
from typing import List, Dict, Tuple

ROOM_CODE = {
    'common_room': 1,
    'master_room': 2,
    'living_room': 3,
    'balcony': 4,
    'bathroom': 5,
    'kitchen': 6,
    'storage': 7,
    'dining': 8,
}


FEATURES_LIST = [
    *(f"uni_{i}_{m}" for i in range(1,7) for m in ['aspect','count','total_area','avg_x','avg_y']),
    *(f"bi_{i}_{j}_{m}" for i in range(1,7) for j in range(i+1,7) for m in ['count','area_ratio','avg_x','avg_y']),
    *(f"tri_{i}_{j}_{k}_{m}" for i in range(1,7) for j in range(i+1,7) for k in range(j+1,7)
      for m in ['count','area_ratio_ij','area_ratio_jk','avg_x','avg_y'])
]

ROOM_NAME_BY_CODE = {
    1: 'common room', 2: 'master bedroom', 3: 'living room', 4: 'balcony',
    5: 'bathroom', 6: 'kitchen', 7: 'storage', 8: 'dining area',
}


class FeatureMapping:
    clustering_to_concept: Dict[str, str] = None
    concept_categories: Dict[str, List[str]] = None

    def __post_init__(self):
        mapping = {}
        for feat in FEATURES_LIST:
            parts = feat.split('_')
            if parts[0] == 'uni':
                _, code, metric = parts
                room = ROOM_NAME_BY_CODE[int(code)]
                desc_map = {
                    'aspect': 'aspect ratio', 'count': 'count', 'total_area': 'total area',
                    'avg_x': 'horizontal position', 'avg_y': 'vertical position'
                }
                mapping[feat] = f"{room} {desc_map[metric]}"
            elif parts[0] == 'bi':
                _, i, j, metric = parts
                room_i = ROOM_NAME_BY_CODE[int(i)]; room_j = ROOM_NAME_BY_CODE[int(j)]
                desc_map = {'count':'frequency','area_ratio':'area ratio',
                            'avg_x':'average horizontal adjacency position','avg_y':'average vertical adjacency position'}
                if metric == 'count': mapping[feat] = f"{room_i} adjacent to {room_j} {desc_map[metric]}"
                elif metric == 'area_ratio': mapping[feat] = f"{room_i} to {room_j} {desc_map[metric]}"
                else: mapping[feat] = f"{room_i}-{room_j} {desc_map[metric]}"
            elif parts[0] == 'tri':
                _, i, j, k, metric = parts
                room_i = ROOM_NAME_BY_CODE[int(i)]; room_j = ROOM_NAME_BY_CODE[int(j)]; room_k = ROOM_NAME_BY_CODE[int(k)]
                desc_map = {
                    'count': 'sequence frequency',
                    'area_ratio_ij': 'to-room area ratio within triple',
                    'area_ratio_jk': 'to-room area ratio within triple',
                    'avg_x': 'average horizontal position', 'avg_y': 'average vertical position'
                }
                if metric == 'count': mapping[feat] = f"{room_i}-{room_j}-{room_k} {desc_map[metric]}"
                elif metric.startswith('area_ratio'): mapping[feat] = f"{room_i} to {room_j if 'ij' in metric else room_k} {desc_map[metric]}"
                else: mapping[feat] = f"{room_i}-{room_j}-{room_k} {desc_map[metric]}"
        self.clustering_to_concept = mapping
        self.concept_categories = {
            'spatial_arrangement': ['position','horizontal','vertical','location'],
            'size_relationships': ['area','ratio','total area'],
            'room_adjacency': ['adjacent','next to','connected'],
            'shape_characteristics': ['aspect ratio','elongated','square'],
            'sequential_patterns': ['sequence','chain','path']
        }