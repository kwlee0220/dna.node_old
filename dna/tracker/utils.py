from typing import List, Optional, Tuple, Sequence, Iterable, Generator, TypeVar

import numpy as np
import numpy.typing as npt

from dna.detect import Detection
from dna.tracker import DNASORTParams
from dna.tracker.dna_deepsort.deepsort.track import Track


T = TypeVar("T")

def subtract(list1:Iterable, list2:List) -> List:
    return [v for v in list1 if v not in list2]

def intersection(list1:Iterable, list2:List) -> List:
    return [v for v in list1 if v in list2]

def all_indices(values:Sequence):
    return list(range(len(values)))

def project(tuples: Iterable[Tuple], elm_idx: int) -> List:
    return [t[elm_idx] for t in tuples]

def get_items(values:Iterable[T], idxes:Iterable[int]) -> Generator[T,None,None]:
    return (values[idx] for idx in idxes)

def get_indexed_items(values:Iterable[T], idxes:Iterable[int]) -> Generator[T,None,None]:
    return ((idx, values[idx]) for idx in idxes)



# def unmatched_track_idxes(tracks:List[Track], matches:List[Tuple[int,int]], track_idxes:Optional[List[int]]=None) -> List[int]:
#     matched_track_idxes = set(project(matches, 0))
#     if track_idxes:
#         return [idx for idx in track_idxes if idx not in matched_track_idxes]
#     else:
#         return [idx for idx, _ in enumerate(tracks) if idx not in matched_track_idxes]

# def hot_track_idxes(tracks:List[Track], matches:List[Tuple[int,int]], track_idxes:Optional[List[int]]=None) -> List[int]:
#     matched_track_idxes = set(project(matches, 0))
#     track_ids = (i for i in track_idxes) if track_idxes else (i for i in range(len(tracks)))
#     return [i for i in track_ids \
#                 if i not in matched_track_idxes and (t:=tracks[i]).is_confirmed() and t.time_since_update <= 3]

# def tentative_track_idxes(tracks:List[Track], matches:List[Tuple[int,int]], track_idxes:Optional[List[int]]=None) -> List[int]:
#     matched_track_idxes = set(project(matches, 0))
#     track_ids = (i for i in track_idxes) if track_idxes else (i for i in range(len(tracks)))
#     return  [i for i in track_ids \
#                     if i not in matched_track_idxes and not tracks[i].is_confirmed()]

# def unmatched_det_idxes(detections:List[Detection], matches:List[Tuple[int,int]], det_idxes:Optional[List[int]]=None) -> List[int]:
#     matched_det_idxes = set(project(matches, 1))
#     det_ids = (i for i in det_idxes) if det_idxes else (i for i in range(len(detections)))
#     return [i for i in det_ids if i not in matched_det_idxes]

# def strong_det_idxes(detections:List[Detection], matches:List[Tuple[int,int]], detection_threshold:float,
#                      det_idxes:Optional[List[int]]=None) -> List[int]:
#     matched_det_idxes = set(project(matches, 1))
#     det_ids = (i for i in det_idxes) if det_idxes else (i for i in range(len(detections)))
#     return [i for i in det_ids if i not in matched_det_idxes and detections[i].score >= detection_threshold]

# def weak_det_idxes(detections:List[Detection], matches:List[Tuple[int,int]], detection_threshold:float,
#                      det_idxes:Optional[List[int]]=None) -> List[int]:
#     matched_det_idxes = set(project(matches, 1))
#     det_ids = (i for i in det_idxes) if det_idxes else (i for i in range(len(detections)))
#     return [i for i in det_ids if i not in matched_det_idxes and detections[i].score < detection_threshold]
    
# def metric_det_idxes(detections:List[Detection], matches:List[Tuple[int,int]], params:DNASORTParams,
#                      det_idxes:Optional[List[int]]=None) -> List[int]:
#     matched_det_idxes = set(project(matches, 1))
#     det_ids = (i for i in det_idxes) if det_idxes else (i for i in range(len(detections)))
#     return [i for i in det_ids if i not in matched_det_idxes and params.is_large_detection_for_metric(detections[i])]