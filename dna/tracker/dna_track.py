from __future__ import annotations
from typing import Union, List

import numpy as np
import cv2

from dna import Box, Size2d, Image, BGR, Point, plot_utils, Frame
from dna.detect import Detection
from dna.tracker import ObjectTrack, TrackState
from .kalman_filter import KalmanFilter
from .dna_track_params import DNATrackParams


def to_tlbr(xyah:np.ndarray) -> Box:
    ret = xyah.copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    ret[2:] = ret[:2] + ret[2:]

    return ret

def to_box(tlbr:np.ndarray) -> Box:
    box = Box.from_tlbr(tlbr)
    if not box.is_valid():
        tl = box.top_left()
        br = tl + Size2d(0.00001, 0.00001)
        box = Box.from_points(tl, br)
    return box


from dataclasses import dataclass, field
@dataclass(frozen=True, eq=True)    # slots=True
class DNATrackState:
    mean: np.ndarray
    covariance: np.ndarray
    hits: int
    time_since_update: int
    detections: List[Detection]
    features: List[np.ndarray]
    frame_index: int
    timestamp: float



class DNATrack(ObjectTrack):
    def __init__(self, mean, covariance, track_id:int, frame_index:int, ts:float,
                    params:DNATrackParams, detection:Detection) -> None:
        super().__init__(id=track_id, state=TrackState.Tentative, location=to_box(to_tlbr(mean[:4])),
                        frame_index=frame_index, timestamp=ts)

        self.mean = mean
        self.covariance = covariance
        self.hits = 1
        self.time_since_update = 0
        self.time_to_promote = params.n_init - 1

        self.archived_state = None

        self.detections = [detection]
        self.features = []
        if detection.feature is not None:
            self.features.append(detection.feature)
        self.home_zone = params.find_stable_zone(detection.bbox)

        self.n_init = params.n_init
        self.max_age = params.max_age
        self.exit_zone_touch_count = 0

    @property
    def age(self) -> int:
        return len(self.detections)

    @property
    def last_detection(self) -> Detection:
        return self.detections[-1]

    def predict(self, kf) -> None:
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.location = to_box(to_tlbr(self.mean[:4]))
        self.time_since_update += 1

    def update(self, kf, frame:Frame, det: Union[Detection, None], track_params:DNATrackParams) -> None:
        self.mean, self.covariance = kf.update(self.mean, self.covariance, det.bbox.to_xyah())
        self.location = to_box(to_tlbr(self.mean[:4]))
        
        self.detections.append(det)
        if det and track_params.is_large_detection_for_metric_regitry(det):
            self.features.append(det.feature)
            if len(self.features) > track_params.max_feature_count:
                self.features = self.features[-track_params.max_feature_count:]
        self.hits += 1
        self.time_since_update = 0
        self.frame_index = frame.index
        self.timestamp = frame.ts
        self.archived_state = None

        if self.state == TrackState.Tentative:
            if det is None:
                self.mark_deleted()
            elif track_params.is_strong_detection(det):
                self.time_to_promote -= 1
                if self.time_to_promote <= 0:
                    self.state = TrackState.Confirmed
            else:
                if self.hits - (self.n_init-self.time_to_promote) > 2:
                    self.mark_deleted()
        elif self.state == TrackState.TemporarilyLost:
            self.state = TrackState.Confirmed
                
    def mark_missed(self, frame:Frame) -> None:
        self.detections.append(None)
        if self.state == TrackState.Tentative:
            self.mark_deleted()
        elif self.time_since_update > self.max_age:
            self.mark_deleted()
        elif self.state != TrackState.TemporarilyLost:
            self.state = TrackState.TemporarilyLost
            self.archived_state = DNATrackState(self.mean, self.covariance, self.hits, self.time_since_update,
                                                self.detections.copy(), self.features.copy(),
                                                self.frame_index, self.timestamp)

    def mark_deleted(self) -> None:
        self.state = TrackState.Deleted

    def mark_exit_zone_touched(self) -> None:
        self.exit_zone_touch_count += 1
        if self.exit_zone_touch_count >= 3:
            self.mark_deleted()
        
    @property
    def state_str(self) -> str:
        if self.state == TrackState.Confirmed:
            return f'{self.id}(C)'
        elif self.state == TrackState.TemporarilyLost:
            return f'{self.id}({self.time_since_update})'
        elif self.state == TrackState.Tentative:
            return f'{self.id}(-{self.time_to_promote})'
        elif self.state == TrackState.Deleted:
            return f'{self.id}(D)'
        else:
            raise ValueError("Shold not be here")
        
    def __repr__(self) -> str:
        millis = int(round(self.timestamp * 1000))
        return (f'{self.state_str}, location={self.location}, age={self.age}({self.time_since_update}), nfeats={len(self.features)}, '
                f'frame={self.frame_index}, ts={millis}')

    def take_over(self, track:DNATrack, kf:KalmanFilter, frame:Frame, track_params:DNATrackParams) -> None:
        archived_state = self.archived_state

        self.mean = archived_state.mean
        self.covariance = archived_state.covariance
        self.time_since_update = archived_state.time_since_update
        self.archived_state = None

        for i in range(archived_state.frame_index, track.first_frame_index):
            self.predict(kf)
            self.detections.append(None)

        for det in track.detections:
            self.predict(kf)
            if det:
                self.update(kf, frame, det, track_params)
            else:
                self.detections.append(None)

        track.mark_deleted()