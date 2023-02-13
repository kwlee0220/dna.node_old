# vim: expandtab:ts=4:sw=4

import numpy as np
import cv2

import dna
from dna import Box, Size2d, Image, BGR, Point, plot_utils, Frame
from dna.detect import Detection
from dna.tracker import DNASORTParams, ObjectTrack, TrackState


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


class DNATrack(ObjectTrack):
    def __init__(self, mean, covariance, track_id:int, frame_index:int, ts:float,
                    n_init:int, max_age:int, detection:Detection=None) -> None:
        super().__init__(id=track_id, state=TrackState.Tentative, location=to_box(to_tlbr(mean[:4])),
                        frame_index=frame_index, timestamp=ts)

        self.mean = mean
        self.covariance = covariance
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        self.features = []
        self.last_detection = detection
        if detection is not None and detection.feature is not None:
            self.features.append(detection.feature)

        self.n_init = n_init
        self.time_to_promote = n_init - 1
        self._max_age = max_age

    def to_tlwh(self) -> np.ndarray:
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self) -> np.ndarray:
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf) -> None:
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.location = Box.from_tlbr(self.to_tlbr())
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, frame:Frame, det: Detection, track_params:DNASORTParams) -> None:
        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.Tentative:
            if track_params.is_strong_detection(det):
                self.time_to_promote -= 1
                if self.time_to_promote <= 0:
                    self.state = TrackState.Confirmed
            else:
                if self.hits - (self.n_init-self.time_to_promote) > 2:
                    self.state = TrackState.Deleted
                    return

        self.mean, self.covariance = kf.update(self.mean, self.covariance, det.bbox.to_xyah())
        self.location = to_box(self.to_tlbr())
        
        if track_params.is_large_detection_for_metric(det):
            self.features.append(det.feature)
            if len(self.features) > track_params.max_feature_count:
                self.features = self.features[-track_params.max_feature_count:]
        self.last_detection = det
        self.frame_index = frame.index
        self.timestamp = frame.ts
                
    def mark_missed(self) -> None:
        if self.state == TrackState.Tentative:
            self.mark_deleted()
        elif self.time_since_update > self._max_age:
            self.mark_deleted()
        else:
            self.state = TrackState.TemporarilyLost
        self.last_detection = None

    def mark_deleted(self) -> None:
        self.state = TrackState.Deleted
        self.last_detection = None
        
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
        return (f'{self.state_str}, location={self.location}, age={self.age}, '
                f'frame={self.frame_index}, ts={millis}')