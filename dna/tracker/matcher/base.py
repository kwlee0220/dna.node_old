
from __future__ import annotations
from typing import Tuple, Iterable, Optional, List

from dna.tracker import utils

INVALID_DIST_DISTANCE = 9999
INVALID_IOU_DISTANCE = 1
INVALID_METRIC_DISTANCE = 1


def matches_str(tracks, matches):
    return ",".join([f'{match_str(tracks, m)}' for m in matches])

def match_str(tracks, match):
    return f'({tracks[match[0]].track_id}, {match[1]})'


from abc import ABCMeta, abstractmethod
class Matcher(metaclass=ABCMeta):
    @abstractmethod
    def match(self, track_idxes:Iterable[int], det_idxes:Iterable[int]) -> List[Tuple[int,int]]:
        pass


class ChainedMatcher(Matcher):
    def __init__(self, *matchers):
        self.matchers = matchers
        
    def match(self, track_idxes:Iterable[int], det_idxes:Iterable[int]) -> List[Tuple[int,int]]:
        unmatched_track_idxes = track_idxes.copy()
        unmatched_det_idxes = det_idxes.copy()

        matches = []
        for matcher in self.matchers:
            if not (unmatched_track_idxes and unmatched_det_idxes):
                break
            
            matches0 = matcher.match(unmatched_track_idxes, unmatched_det_idxes)
            if matches0:
                matches += matches0
                unmatched_track_idxes = utils.subtract(unmatched_track_idxes, utils.project(matches0, 0))
                unmatched_det_idxes = utils.subtract(unmatched_det_idxes, utils.project(matches0, 1))
        return matches

    def __repr__(self) -> str:
        return '+'.join([repr(m) for m in self.matchers])
    

def chain(*matchers) -> Matcher:
    return ChainedMatcher(*matchers)

from dna.tracker.utils import DNASORTParams
class MatchingSession:
    def __init__(self, tracks, detections, params:DNASORTParams, track_idxes=None, det_idxes=None) -> None:
        self.tracks = tracks
        self.detections = detections
        self.params = params
        self.matches = []
        self.unmatched_track_idxes = track_idxes.copy() if track_idxes else utils.all_indices(tracks)
        self.unmatched_det_idxes = det_idxes.copy() if det_idxes else utils.all_indices(detections)

    def update(self, matches0:List[Tuple[int,int]]) -> None:
        self.matches += matches0
        self.unmatched_track_idxes = utils.subtract(self.unmatched_track_idxes, utils.project(matches0, 0))
        self.unmatched_det_idxes = utils.subtract(self.unmatched_det_idxes, utils.project(matches0, 1))

    def remove_det_idxes(self, idxes:List[int]) -> None:
        self.unmatched_det_idxes = utils.subtract(self.unmatched_det_idxes, idxes)

    def pull_out(self, match:Tuple[int,int]) -> None:
        self.matches = [m for m in self.matches if m != match]
        self.unmatched_track_idxes.append(match[0])
        self.unmatched_det_idxes.append(match[1])

    def find_match_by_track(self, track_idx:int) -> Optional[Tuple[int,int]]:
        founds = [m for m in self.matches if m[0] == track_idx]
        return founds[0] if founds else None

    def find_match_by_det(self, det_idx:int) -> Optional[Tuple[int,int]]:
        founds = [m for m in self.matches if m[1] == det_idx]
        return founds[0] if founds else None

    @property
    def unmatched_hot_track_idxes(self) -> List[int]:
        idx_tracks = ((i, self.tracks[i]) for i in self.unmatched_track_idxes)
        return [i for i, t in idx_tracks if t.is_confirmed() and t.time_since_update <= 3]

    @property
    def unmatched_tlost_track_idxes(self) -> List[int]:
        idx_tracks = ((i, self.tracks[i]) for i in self.unmatched_track_idxes)
        return [i for i, t in idx_tracks if t.is_confirmed() and t.time_since_update > 3]

    @property
    def unmatched_tentative_track_idxes(self) -> List[int]:
        return [i for i in self.unmatched_track_idxes if not self.tracks[i].is_confirmed()]
    @property
    def unmatched_confirmed_track_idxes(self) -> List[int]:
        return [i for i in self.unmatched_track_idxes if self.tracks[i].is_confirmed()]
        
    @property
    def unmatched_strong_det_idxes(self) -> List[int]:
        return [i for i in self.unmatched_det_idxes if self.params.is_strong_detection(self.detections[i])]
        
    @property
    def unmatched_weak_det_idxes(self) -> List[int]:
        return [i for i in self.unmatched_det_idxes if not self.params.is_strong_detection(self.detections[i])]
        
    @property
    def unmatched_metric_det_idxes(self) -> List[int]:
        return [i for i in self.unmatched_det_idxes if self.params.is_large_detection_for_metric(self.detections[i])]

    def __repr__(self) -> str:
        bindings = [(self.tracks[t_idx].track_id, d_idx) for t_idx, d_idx in self.matches]
        um_track_idxes = [self.tracks[t_idx].track_id for t_idx in self.unmatched_track_idxes]
        return (f'matches={bindings}, unmatched: tracks={um_track_idxes}, det_idxes={self.unmatched_det_idxes}')
    
         