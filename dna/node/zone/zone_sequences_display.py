from __future__ import annotations

from typing import Optional, Union
from collections import defaultdict
import heapq

import cv2

from dna import Frame, Box, color
from dna.camera import FrameProcessor, ImageProcessor
from dna.track import TrackState
from dna.event import TrackEvent, TrackletMotion, EventQueue, EventListener
from .types import ResidentChanged
from .resident_changes import ResidentChanges


class ZoneSequenceDisplay(FrameProcessor,EventListener):
    def __init__(self, motion_definitions:dict[str,str]) -> None:
        self.motion_counts:dict[str,int] = defaultdict(int)
        for motion_id in motion_definitions.values():
            self.motion_counts[motion_id] = 0
        self.track_locations:dict[str,Box] = dict()
        self.motion_tracks:set[int] = set()

    def close(self) -> None:
        for key in self.motion_counts.keys():
            self.motion_counts[key] = 0
        
    def handle_event(self, ev:Union[TrackEvent,TrackletMotion]) -> None:
        if isinstance(ev, TrackEvent):
            if ev.state == TrackState.Deleted:
                self.track_locations[ev.track_id] = ev.location
        elif isinstance(ev, TrackletMotion) and ev.motion:
            self.motion_counts[ev.motion] += 1
            self.motion_tracks.add(ev.track_id)

    def on_started(self, proc:ImageProcessor) -> None:
        for key in self.motion_counts.keys():
            self.motion_counts[key] = 0

    def on_stopped(self) -> None:
        pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        y_offset = 20
        convas = frame.image
        
        for track_id, loc in self.track_locations.items():
            if track_id in self.motion_tracks:
                convas = loc.draw(convas, color.RED, line_thickness=3)
        self.track_locations.clear()
        self.motion_tracks.clear()

        for motion, count in self.motion_counts.items():
            y_offset += 25
            convas = cv2.putText(convas, f'motion={motion:>3}, count={count}',
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.RED, 2)
        return Frame(image=convas, index=frame.index, ts=frame.ts)

    def set_control(self, key:int) -> int:
        return key