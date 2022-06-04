from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from contextlib import suppress
from pathlib import Path
import logging

import numpy as np
import cv2
from omegaconf import OmegaConf

import dna
from dna import Box, Image, plot_utils, BGR, Frame
from dna.camera import Camera, ImageProcessor
from dna.detect import Detection
from dna.detect.object_detector import ObjectDetector


from enum import Enum
class TrackState(Enum):
    Null = 0
    Tentative = 1
    Confirmed = 2
    TemporarilyLost = 3
    Deleted = 4


@dataclass(frozen=True, eq=True, order=True)    # slots=True
class Track:
    id: int = field(compare=False)
    state: TrackState = field(compare=False)
    location: Box = field(compare=False)
    frame_index: int = field(compare=True)
    ts: float = field(compare=False)

    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_temporarily_lost(self) -> bool:
        return self.state == TrackState.TemporarilyLost

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted
    
    def __repr__(self) -> str:
        epoch = int(round(self.ts * 1000))
        return f"{self.state.name}[{self.id}]={self.location}, frame={self.frame_index}, ts={epoch}"

    def draw(self, convas:Image, color:BGR, label_color=None, line_thickness=2) -> Image:
        loc = self.location

        convas = loc.draw(convas, color, line_thickness=line_thickness)
        convas = cv2.circle(convas, loc.center().xy.astype(int), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        if label_color:
            convas = plot_utils.draw_label(convas, str(self.id), loc.tl.astype(int), label_color, color, 2)

        return convas

    def to_string(self) -> str:
        tlbr = self.location.to_tlbr()
        epoch = int(round(self.ts * 1000))
        return (f"{self.frame_index},{self.id},{tlbr[0]:.0f},{tlbr[1]:.0f},{tlbr[2]:.0f},{tlbr[3]:.0f},"
                f"{self.state.name},{epoch}")
    
    @staticmethod
    def from_string(csv) -> Track:
        parts = csv.split(',')

        frame_idx = int(parts[0])
        track_id = int(parts[1])
        tlbr = np.array(parts[2:6]).astype(int)
        bbox = Box.from_tlbr(tlbr)
        state = TrackState(int(parts[6]))
        ts = int(parts[7]) / 1000
        
        return Track(id=track_id, state=state, location=bbox, frame_index=frame_idx, ts=ts)


class ObjectTracker(metaclass=ABCMeta):
    logger = logging.getLogger("dna.track.tracker")
    logger.setLevel(logging.INFO)

    @abstractmethod
    def track(self, frame: Frame) -> List[Track]: pass


class TrackerCallback(metaclass=ABCMeta):
    @abstractmethod
    def track_started(self, tracker:ObjectTracker) -> None: pass

    @abstractmethod
    def track_stopped(self, tracker:ObjectTracker) -> None: pass

    @abstractmethod
    def tracked(self, tracker: ObjectTracker, frame: Frame, tracks: List[Track]) -> None: pass


class DetectionBasedObjectTracker(ObjectTracker):
    @property
    @abstractmethod
    def detector(self) -> ObjectDetector: pass

    @abstractmethod
    def last_frame_detections(self) -> List[Detection]: pass


class LogFileBasedObjectTracker(ObjectTracker):
    def __init__(self, track_file: Path) -> None:
        """[Create an ObjectTracker object that issues tracking events from a tracking log file.]

        Args:
            det_file (Path): Path to the detection file.
        """
        self.__file = open(track_file, 'r')
        self.look_ahead = self._look_ahead()

    @property
    def file(self) -> Path:
        return self.__file

    def track(self, frame: Frame) -> List[Track]:
        if not self.look_ahead:
            return []

        if self.look_ahead.frame_index > frame.index:
            return []

        # throw track event lines upto target_idx -
        while self.look_ahead.frame_index < frame.index:
            self.look_ahead = self._look_ahead()

        tracks = []
        while self.look_ahead.frame_index == frame.index:
            tracks.append(self.look_ahead)

            # read next line
            self.look_ahead = self._look_ahead()
            if self.look_ahead is None:
                break

        return tracks
        
    def _look_ahead(self) -> Track:
        line = self.__file.readline().rstrip()
        if line:
            return Track.from_string(line)
        else:
            self.__file.close()
            return None

    def __repr__(self) -> str:
        current_idx = int(self.look_ahead[0]) if self.look_ahead else -1
        return f"{self.__class__.__name__}: frame_idx={current_idx}, from={self.__file.name}"