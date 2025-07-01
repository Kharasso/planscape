import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re
from collections import defaultdict
import asyncio
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
