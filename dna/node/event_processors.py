from __future__ import annotations
import sys
from typing import List, Dict

from collections import defaultdict
from pathlib import Path

from .track_event import TrackEvent
from .event_processor import EventProcessor


class GroupByFrameIndex(EventProcessor):
    def __init__(self, max_delay_frames:int) -> None:
        EventProcessor.__init__(self)

        self.groups:Dict[int,List[TrackEvent]] = defaultdict(list)
        self.max_delay_frames = max_delay_frames
    
    def close(self) -> None:
        while self.groups:
            min_frame_index = min(self.groups.keys())
            group = self.groups.pop(min_frame_index, None)
            self.publish_event(group)
        super().close()

    def handle_event(self, ev:TrackEvent) -> None:
        group = self.groups[ev.frame_index]
        group.append(ev)
        
        if len(self.groups) > self.max_delay_frames:
            min_frame_index = min(self.groups.keys())
            group = self.groups.pop(min_frame_index, None)
            self.publish_event(group)
            
class UngroupTrackEvents(EventProcessor):
    def handle_event(self, group: List[TrackEvent]) -> None:
        for ev in group:
            self.publish_event(ev)