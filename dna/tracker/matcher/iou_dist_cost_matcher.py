from __future__ import annotations
from typing import Tuple, Iterable, List, Optional

import logging
import numpy as np
import numpy.typing as npt

import dna
from dna import Box
from dna.detect import Detection
from dna.tracker import utils
from dna.tracker.matcher import MatchingSession, matches_str
from ..dna_track_params import DNATrackParams
from .base import Matcher, chain
from .reciprocal_cost_matcher2 import ReciprocalCostMatcher


class IoUDistanceCostMatcher(Matcher):
    def __init__(self, tracks:List, detections:List[Detection],
                 params:DNATrackParams,
                 iou_cost:np.ndarray, dist_cost:np.ndarray,
                 logger:Optional[logging.Logger]=None) -> None:
        self.tracks = tracks
        self.detections = detections
        self.params = params
        self.iou_cost = iou_cost
        self.dist_cost = dist_cost
        self.logger = logger

        IOU_DIST_THRESHOLD = params.iou_dist_threshold
        dist_matcher = ReciprocalCostMatcher(self.dist_cost, IOU_DIST_THRESHOLD.distance, name='dist', logger=self.logger)
        iou_matcher = ReciprocalCostMatcher(self.iou_cost, IOU_DIST_THRESHOLD.iou, name='iou', logger=self.logger)
        self.matcher = chain(dist_matcher, iou_matcher)

    def match(self, track_idxes:Iterable[int], det_idxes:Iterable[int]) -> List[Tuple[int,int]]:
        session = MatchingSession(self.tracks, self.detections, self.params, track_idxes, det_idxes)
            
        ###########################################################################################################
        ### 일반 threshold를 기준으로 distance와 IoU 거리 정보를 사용해서 matching 실시.
        ###########################################################################################################
        if session.unmatched_hot_track_idxes and session.unmatched_det_idxes:
            matches0 = self.matcher.match(session.unmatched_hot_track_idxes, session.unmatched_det_idxes)
            if matches0:
                session.update(matches0)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'hot, det_all, {self}: {matches_str(self.tracks, matches0)}')
        if session.unmatched_tentative_track_idxes and session.unmatched_det_idxes:
            matches0 = self.matcher.match(session.unmatched_tentative_track_idxes, session.unmatched_det_idxes)
            if matches0:
                session.update(matches0)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'tentative, det_all, {self}: {matches_str(self.tracks, matches0)}')

        return session.matches

    def __repr__(self) -> str:
        return repr(self.matcher)