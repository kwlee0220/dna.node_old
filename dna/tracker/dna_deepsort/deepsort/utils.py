from collections import namedtuple
from typing import List, Union, Tuple

import numpy as np

from dna import Box, Size2d

def all_indices(values):
    return list(range(len(values)))

def intersection(list1, list2):
    return [v for v in list1 if v in list2]

def subtract(list1, list2):
    return [v for v in list1 if v not in list2]

def track_to_box(track, epsilon=0.00001):
    box = Box.from_tlbr(track.to_tlbr())
    if not box.is_valid():
        tl = box.top_left()
        br = tl + Size2d(epsilon, epsilon)
        box = Box.from_points(tl, br)
    return box

def boxes_distance(tlbr1, tlbr2):
    delta1 = tlbr1[0,3] - tlbr2[2,1]
    delta2 = tlbr2[0,3] - tlbr2[2,1]
    u = np.max(np.array([np.zeros(len(delta1)), delta1]), axis=0)
    v = np.max(np.array([np.zeros(len(delta2)), delta2]), axis=0)
    dist = np.linalg.norm(np.concatenate([u, v]))
    return dist

def overlap_ratios(box1, box2) -> Tuple[float,float,float]:
    inter_area = box1.intersection(box2).area()
    r1 = inter_area / box1.area() if box1.is_valid() else 0
    r2 = inter_area / box2.area() if box2.is_valid() else 0
    iou = inter_area / (box1.area() + box2.area() - inter_area)  if box1.is_valid() and box2.is_valid() else 0
    return (r1, r2, iou)

def overlaps(box, candidate_boxes, candidate_indices=None) -> List[Tuple[int,Tuple[float,float,float]]]:
    if not candidate_indices:
        candidate_indices = list(range(len(candidate_boxes)))
    return [(idx, overlap_ratios(box, candidate_boxes[idx])) for idx in candidate_indices]

def overlaps_threshold(box, candidate_boxes, threshold, candidate_indices=None) -> List[Tuple[int,float]]:
    return [(idx, ov) for idx, ov in overlaps(box, candidate_boxes, candidate_indices) if max(ov) >= threshold]

def overlaps_cond(box, candidate_boxes, filter, candidate_indices=None) -> List[Tuple[int,float]]:
    return [(idx, ov) for idx, ov in overlaps(box, candidate_boxes, candidate_indices) if filter(ov)]

def split_tuples(tuples: List[Tuple]):
    firsts = []
    seconds = []
    for t in tuples:
        firsts.append(t[0])
        seconds.append(t[1])

    return firsts, seconds

def project(tuples: List[Tuple], idx: int):
    return [t[idx] for t in tuples]