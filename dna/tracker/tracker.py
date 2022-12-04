from __future__ import annotations

from typing import List, Optional, Any
from datetime import datetime
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from contextlib import suppress
from pathlib import Path

import numpy as np
import cv2

from dna import Frame
from dna.detect import Detection
from dna.detect.object_detector import ObjectDetector
from .dna_track import DNATrack

class ObjectTracker(metaclass=ABCMeta):
    @abstractmethod
    def track(self, frame: Frame) -> List[DNATrack]: pass


class TrackProcessor(metaclass=ABCMeta):
    @abstractmethod
    def track_started(self, tracker:ObjectTracker) -> None: pass

    @abstractmethod
    def track_stopped(self, tracker:ObjectTracker) -> None: pass

    @abstractmethod
    def process_tracks(self, tracker: ObjectTracker, frame: Frame, tracks: List[DNATrack]) -> None: pass


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

    def track(self, frame: Frame) -> List[DNATrack]:
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
        
    def _look_ahead(self) -> DNATrack:
        line = self.__file.readline().rstrip()
        if line:
            return DNATrack.from_string(line)
        else:
            self.__file.close()
            return None

    def __repr__(self) -> str:
        current_idx = int(self.look_ahead[0]) if self.look_ahead else -1
        return f"{self.__class__.__name__}: frame_idx={current_idx}, from={self.__file.name}"