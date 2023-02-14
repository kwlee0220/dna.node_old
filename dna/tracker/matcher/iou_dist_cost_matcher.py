from __future__ import annotations
from typing import Tuple, Iterable, List, Optional

import logging
import numpy as np
import numpy.typing as npt

import dna
from dna import Box
from dna.detect import Detection
from dna.tracker import ObjectTrack, utils
from dna.tracker.matcher import MatchingSession, matches_str
from ..dna_track_params import DNATrackParams
from .base import Matcher, chain
from .reciprocal_cost_matcher import ReciprocalCostMatcher


class IoUDistanceCostMatcher(Matcher):
    def __init__(self, tracks:List[ObjectTrack], detections:List[Detection],
                 params:DNATrackParams,
                 iou_cost:np.ndarray, dist_cost:np.ndarray,
                 logger:Optional[logging.Logger]=None) -> None:
        self.tracks = tracks
        self.detections = detections
        self.params = params
        self.logger = logger

        IOU_DIST_THRESHOLD = params.iou_dist_threshold
        dist_matcher = ReciprocalCostMatcher(dist_cost, IOU_DIST_THRESHOLD.distance, name='dist', logger=self.logger)
        iou_matcher = ReciprocalCostMatcher(iou_cost, IOU_DIST_THRESHOLD.iou, name='iou', logger=self.logger)
        self.matcher = chain(dist_matcher, iou_matcher)

    def match(self, track_idxes:Iterable[int], det_idxes:Iterable[int]) -> List[Tuple[int,int]]:
        session = MatchingSession(self.tracks, self.detections, self.params, track_idxes, det_idxes)

        #   
        # Do matching between a set of tracks and a set of detections based on motion information (distance and IoU).
        # The matching is performed on hot tracks and tentative tracks only. The tracks that are unmatched more
        # than 2 times are left out in this matching, because we are not quite sure of their estimated locations.
        #

        # Give preference to hot tracks in matching
        if session.unmatched_hot_track_idxes and session.unmatched_det_idxes:
            matches0 = self.matcher.match(session.unmatched_hot_track_idxes, session.unmatched_det_idxes)
            if matches0:
                session.update(matches0)
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'hot, det_all, {self}: {matches_str(self.tracks, matches0)}')

        if session.unmatched_tentative_track_idxes and session.unmatched_det_idxes:
            matches0 = self.matcher.match(session.unmatched_tentative_track_idxes, session.unmatched_det_idxes)
            if matches0:
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'tentative, det_all, {self}: {matches_str(self.tracks, matches0)}')
                session.update(matches0)

        return session.matches

    def __repr__(self) -> str:
        return repr(self.matcher)