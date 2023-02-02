from typing import List, Tuple, Sequence, Iterable, Generator

import numpy as np

from dna import Box, Size2d
from .track import Track

def all_indices(values:Sequence):
    return list(range(len(values)))

def intersection(coll1:Iterable, coll2:List) -> List:
    return [v for v in coll1 if v in coll2]

def subtract(coll1:Iterable, coll2:List) -> List:
    return [v for v in coll1 if v not in coll2]

def project(tuples: Iterable[Tuple], elm_idx: int) -> List:
    return [t[elm_idx] for t in tuples]

def get_items(values:List, idxes:Iterable[int]) -> Generator:
    return (values[idx] for idx in idxes)

def get_indexed_items(values:List, idxes:Iterable[int]) -> Generator:
    return ((idx, values[idx]) for idx in idxes)

def track_to_box(track:Track, epsilon:float=0.00001) -> Box:
    box = Box.from_tlbr(track.to_tlbr())
    if not box.is_valid():
        tl = box.top_left()
        br = tl + Size2d(epsilon, epsilon)
        box = Box.from_points(tl, br)
    return box

def overlap_boxes(target:Box, boxes:List[Box], box_indices:List[int]=None) \
    -> List[Tuple[int, Tuple[float,float,float]]]:
    if box_indices is None:
        return ((idx, target.overlap_ratios(box)) for idx, box in enumerate(boxes))
    else:
        return ((idx, target.overlap_ratios(boxes[idx])) for idx in box_indices)