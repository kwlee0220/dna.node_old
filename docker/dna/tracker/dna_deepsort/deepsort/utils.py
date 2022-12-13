from collections import namedtuple
from typing import List, Union, Tuple

import numpy as np

from dna import Box, Size2d

def all_indices(values):
    return list(range(len(values)))

def intersection(coll1, coll2):
    return [v for v in coll1 if v in coll2]

def subtract(coll1, coll2):
    return [v for v in coll1 if v not in coll2]

def track_to_box(track, epsilon=0.00001):
    box = Box.from_tlbr(track.to_tlbr())
    if not box.is_valid():
        tl = box.top_left()
        br = tl + Size2d(epsilon, epsilon)
        box = Box.from_points(tl, br)
    return box

# 'candidate_boxes'에 포함된 box들과 'box' 사이의 겹침 정보를 반환한다.
def _overlaps(box, candidate_boxes, candidate_idxs=None) -> List[Tuple[int,Tuple[float,float,float]]]:
    if candidate_idxs is None:
        candidate_idxs = all_indices(candidate_boxes)
    return [(idx, box.overlap_ratios(candidate_boxes[idx])) for idx in candidate_idxs]

def filter_overlaps(box, candidate_boxes, cond, candidate_idxs=None) -> List[Tuple[int,float]]:
    return [(idx, ov) for idx, ov in _overlaps(box, candidate_boxes, candidate_idxs) if cond(ov)]

def split_tuples(tuples: List[Tuple]):
    firsts = []
    seconds = []
    for t in tuples:
        firsts.append(t[0])
        seconds.append(t[1])

    return firsts, seconds

def project(tuples: List[Tuple], idx: int):
    return [t[idx] for t in tuples]