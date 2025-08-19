
# finetune_qwen2p5_vl_odd_one_out.py
# -----------------------------------------------------------
# Fine-tune Qwen2.5-VL-7B-Instruct with QLoRA on images+text
# to predict the "odd one out" letter (A/B/C/D/E) from
# 5-panel floorplan images using your CSV metadata.
#
# Usage (example):
#   pip install -U "transformers>=4.43.0" accelerate peft bitsandbytes trl datasets pandas pillow evaluate scikit-learn
#   python finetune_qwen2p5_vl_odd_one_out.py \
#       --csv /path/to/easy_dataset_mid_train_test.csv \
#       --image_root /path/to/datasets/easy \
#       --model_name Qwen/Qwen2.5-VL-7B-Instruct \
#       --output_dir ./qwen2p5_vl_odd1out \
#       --num_train_epochs 2 --per_device_train_batch_size 1
#
# Notes:
# - This script performs supervised fine-tuning where the target output is a
#   single letter (A–E). It uses the model's chat template and masks the
#   prompt tokens in the loss.
# - It applies LoRA adapters (QLoRA with 4-bit loading) to keep VRAM usage
#   reasonable. You can adjust target_modules based on your environment/model
#   version (see the comment near the LoraConfig below).
# - For best results ensure you have recent CUDA, PyTorch and Transformers.
# - The CSV must include: split (train/test), orig_index, outlier_id
# - Images are resolved as: {image_root}/example_{orig_index}_2.png
# -----------------------------------------------------------

import os
import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import bitsandbytes as bnb  # noqa: F401 (needed for 4-bit/8-bit quantization)

LETTER_VOCAB = ["A", "B", "C", "D", "E"]


def build_prompt(options: List[str], legend: bool) -> str:
    text = (
        "I am showing you five apartment floorplans, labeled A through E.\n"
        "One of these plans has a different underlying floorplan pattern, while the other four share the same pattern.\n\n"
        "The thick black outline of each of the floorplans indicates the boundary of that floorplan. "
        "The red bar drawn on the black outline of each of the floorplans marks the main entrance of that floorplan.\n"
    )
    if legend:
        text += "There is a color legend indicating the color-coding of room types below all the floorplans.\n"
    text += (
        "Examine each floorplan only within its thick black outer boundary, focusing on spatial layout, room types, and relative sizes.\n"
        "Question: Which floorplan (A, B, C, D, or E) has a different underlying pattern?\n\n"
        "Answer with a single letter only (A/B/C/D/E)."
    )
    return text


class OddOneOutDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, processor: AutoProcessor, legend: bool = False):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.processor = processor
        self.legend = legend

    def __len__(self):
        return len(self.df)

    def _messages(self, image: Image.Image, prompt_text: str, answer_text: str):
        # Qwen2.5-VL chat format: list of {role, content=[{type:"text"/"image", ...}, ...]}
        user_part = [
            {"type": "text", "text": prompt_text},
            {"type": "image", "image": image},
        ]
        assistant_part = [{"type": "text", "text": answer_text}]
        return [
            {"role": "user", "content": user_part},
            {"role": "assistant", "content": assistant_part},
        ]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        # Build path like example_{orig_index}_2.png
        img_name = f"example_{row['orig_index']}_2.png"
        img_path = os.path.join(self.image_root, img_name)
        if not os.path.exists(img_path):
            # Also try absolute path if user placed files differently
            alt_path = os.path.join(self.image_root, "datasets", "easy", img_name)
            if os.path.exists(alt_path):
                img_path = alt_path

        image = Image.open(img_path).convert("RGB")

        prompt_text = build_prompt(LETTER_VOCAB, legend=self.legend)
        answer_text = str(row["outlier_id"]).strip()

        # For efficiency, we pre-create the templated texts and keep the raw PIL image here.
        # Collator will tokenize/pad and create labels with the prompt masked.
        # The 'prompt_only_text' corresponds to user message only, as used for masking.
        messages_full = self._messages(image, prompt_text, answer_text)
        messages_prompt_only = [{"role": "user", "content": messages_full[0]["content"]}]

        # Convert chat templates to plain strings (with special tokens) up-front
        text_full = self.processor.apply_chat_template(messages_full, tokenize=False)
        text_prompt_only = self.processor.apply_chat_template(messages_prompt_only, tokenize=False)

        return {
            "image": image,
            "text_full": text_full,
            "text_prompt": text_prompt_only,
            "label_letter": answer_text,
            "idx": int(row["orig_index"]),
        }


@dataclass
class VLDataCollator:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Unpack
        images = [f["image"] for f in features]
        text_full_list = [f["text_full"] for f in features]
        text_prompt_list = [f["text_prompt"] for f in features]

        # Process the pair (text + images) together to get input_ids, pixel_values, etc.
        batch_inputs = self.processor(
            text=text_full_list,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        tokenizer = self.processor.tokenizer

        # Build labels where prompt tokens are masked (-100)
        labels_list = []
        for tfull, tprompt in zip(text_full_list, text_prompt_list):
            ids_full = tokenizer(tfull, add_special_tokens=False).input_ids
            ids_prompt = tokenizer(tprompt, add_special_tokens=False).input_ids
            lab = ids_full.copy()
            lab[: len(ids_prompt)] = [-100] * len(ids_prompt)
            labels_list.append(torch.tensor(lab, dtype=torch.long))

        # Pad labels to the same length as input_ids
        # Use the tokenizer pad to keep token type consistency
        labels_padded = tokenizer.pad(
            {"input_ids": labels_list},
            padding="longest",
            return_tensors="pt",
        )["input_ids"]

        batch_inputs["labels"] = labels_padded
        return batch_inputs


def get_model_and_processor(model_name: str, bnb_4bit: bool = True):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if bnb_4bit:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            quantization_config=dict(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    # Attach LoRA adapters on the language model submodules.
    # NOTE: You may need to tweak target_modules to match your installed
    # transformers/model version. Common linear projection names are used here.
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # If your model exposes a vision-text projector (e.g., mm_projector),
        # include it in modules_to_save so it remains trainable:
        modules_to_save=None,  # e.g., ["mm_projector"]
    )
    model = get_peft_model(model, lora_cfg)
    return model, processor


def run_eval(model, processor, df_eval: pd.DataFrame, image_root: str, legend: bool, max_samples: int = None):
    model.eval()
    ds = OddOneOutDataset(df_eval if max_samples is None else df_eval.iloc[:max_samples], image_root, processor, legend)
    correct = 0
    total = 0
    preds, gts, idxs = [], [], []

    for i in range(len(ds)):
        ex = ds[i]
        # Build inputs for generation
        inputs = processor(
            text=[ex["text_prompt"]],  # generation prompt (user-only)
            images=[ex["image"]],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                num_beams=1,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        out = processor.tokenizer.decode(gen[0], skip_special_tokens=True)
        # Extract the first valid letter A-E from the generated text
        pred_letter = None
        for ch in out:
            if ch in LETTER_VOCAB:
                pred_letter = ch
                break

        gt = ex["label_letter"]
        if pred_letter == gt:
            correct += 1
        total += 1
        preds.append(pred_letter if pred_letter is not None else "")
        gts.append(gt)
        idxs.append(ex["idx"])

    acc = correct / max(1, total)
    return {"accuracy": acc, "preds": preds, "gts": gts, "idxs": idxs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True,
                        help="Folder that contains example_{orig_index}_2.png files. Example: /path/to/datasets/easy")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./qwen2p5_vl_odd1out")
    parser.add_argument("--legend_in_prompt", action="store_true",
                        help="If provided, includes legend sentence in the prompt.")
    parser.add_argument("--seed", type=int, default=7)

    # Training args
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    df = pd.read_csv(args.csv)
    # Basic sanitation
    assert {"split", "orig_index", "outlier_id"}.issubset(df.columns), \
        "CSV must contain 'split', 'orig_index' and 'outlier_id' columns."

    # Keep only A-E rows
    df = df[df["outlier_id"].isin(LETTER_VOCAB)].copy()

    train_df = df[df["split"].str.lower().isin(["train", "training"])]
    eval_df = df[df["split"].str.lower().isin(["test", "eval", "validation", "val"])]

    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=args.seed)

    # Load model & processor with QLoRA
    model, processor = get_model_and_processor(args.model_name, bnb_4bit=True)

    # Datasets
    train_dataset = OddOneOutDataset(train_df, args.image_root, processor, legend=args.legend_in_prompt)
    eval_dataset = OddOneOutDataset(eval_df, args.image_root, processor, legend=args.legend_in_prompt)

    collator = VLDataCollator(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy,
        bf16=args.bf16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,  # IMPORTANT for VLMs
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # Train
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Quick evaluation pass with greedy decoding
    if len(eval_df) > 0:
        eval_out = run_eval(model, processor, eval_df, args.image_root, args.legend_in_prompt)
        acc = eval_out["accuracy"]
        print(f"[Eval] accuracy={acc:.4f} on {len(eval_out['gts'])} examples")
        with open(os.path.join(args.output_dir, "eval_predictions.jsonl"), "w") as f:
            for idx, p, g in zip(eval_out["idxs"], eval_out["preds"], eval_out["gts"]):
                f.write(json.dumps({"orig_index": idx, "pred": p, "label": g}) + "\n")
        with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
            json.dump({"accuracy": acc, "n": len(eval_out["gts"])}, f, indent=2)

    print("Done.")
    print("Adapter weights are saved under:", args.output_dir)

if __name__ == "__main__":
    main()
