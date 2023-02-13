from __future__ import annotations
from typing import List, Tuple, Set
from abc import ABCMeta, abstractmethod
from enum import Enum
from dataclasses import dataclass, field

import shapely.geometry as geometry
import cv2

from dna import Box, Point, Size2d, Image, BGR, plot_utils, Frame
from dna.detect import Detection

from collections import namedtuple
IouDistThreshold = namedtuple('IouDistThreshold', 'iou,distance')


class TrackState(Enum):
    Null = (0, 'N')
    Tentative = (1, 'T')
    Confirmed = (2, 'C')
    TemporarilyLost = (3, 'L')
    Deleted = (4, 'D')
    
    def __init__(self, code, abbr) -> None:
        super().__init__()
        self.code = code
        self.abbr = abbr


class ObjectTrack:
    def __init__(self, id:int, state:TrackState, location:Box, frame_index:int, timestamp:float) -> None:
        self.id = id
        self.state = state
        self.location = location
        self.frame_index = frame_index
        self.timestamp = timestamp

    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_temporarily_lost(self) -> bool:
        return self.state == TrackState.TemporarilyLost

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted

    def __repr__(self) -> str:
        state_str = f'{self.id}({self.state.abbr})'
        millis = int(round(self.timestamp * 1000))
        return 'f{state_str}, location={self.location}, frame={self.frame_index}, ts={millis}'

    def to_csv(self) -> str:
        t, l, b, r = tuple(self.location.tlbr)
        millis = int(round(self.timestamp * 1000))
        return (f"{self.frame_index},{self.id},{t:.0f},{l:.0f},{b:.0f},{r:.0f},{self.state.name},{millis}")

    def draw(self, convas:Image, color:BGR, label_color:BGR=None, line_thickness:int=2) -> Image:
        loc = self.location
        convas = loc.draw(convas, color, line_thickness=line_thickness)
        convas = cv2.circle(convas, loc.center().xy.astype(int), 2, color, thickness=-1, lineType=cv2.LINE_AA)
        if label_color:
            # label = f"{self.id}({self.state.abbr})"
            label = f"{self.state_str}"
            convas = plot_utils.draw_label(convas, label, Point.from_np(loc.tl.astype(int)),
                                            color=label_color, fill_color=color, thickness=2)
        return convas


class ObjectTracker(metaclass=ABCMeta):
    @abstractmethod
    def track(self, frame: Frame) -> List[ObjectTrack]: pass

    @property
    @abstractmethod
    def tracks(self) -> List[ObjectTrack]: pass


class TrackProcessor(metaclass=ABCMeta):
    @abstractmethod
    def track_started(self, tracker:ObjectTracker) -> None: pass

    @abstractmethod
    def track_stopped(self, tracker:ObjectTracker) -> None: pass

    @abstractmethod
    def process_tracks(self, tracker: ObjectTracker, frame: Frame, tracks: List[ObjectTrack]) -> None: pass


@dataclass(frozen=True, eq=True)    # slots=True
class DNASORTParams:
    detection_classes: Set[str]
    detection_threshold: float
    detection_min_size: Size2d
    detection_max_size: Size2d

    iou_dist_threshold: IouDistThreshold
    iou_dist_threshold_loose: IouDistThreshold

    metric_threshold: float
    metric_threshold_loose: float
    metric_gate_distance: float
    metric_gate_box_distance: float
    metric_min_detection_size: Size2d
    max_feature_count: int

    n_init: int
    min_new_track_size: Size2d
    max_age: int
    overlap_supress_ratio: float

    blind_zones: List[geometry.Polygon]
    exit_zones: List[geometry.Polygon]
    stable_zones: List[geometry.Polygon]

    def is_strong_detection(self, det:Detection) -> bool:
        return det.score >= self.detection_threshold
    
    def is_large_detection_for_metric(self, det:Detection) -> bool:
        return det.bbox.size().width >= self.metric_min_detection_size.width \
                and det.bbox.size().height >= self.metric_min_detection_size.height