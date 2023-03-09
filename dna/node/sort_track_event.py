from __future__ import annotations
import sys
from typing import List, Dict

from collections import defaultdict
from pathlib import Path

from .track_event import TrackEvent
from .event_processor import EventProcessor


class SortTrackEvent(EventProcessor):
    def __init__(self, buffer_size:int) -> None:
        EventProcessor.__init__(self)

        self.buffer_size = buffer_size
        self.buffers:Dict[int,List[TrackEvent]] = defaultdict(list)
        self.min_frame_index = 0
    
    def close(self) -> None:
        while self.buffers:
            self.publish_events_at_frame_index(self.min_frame_index)
            self.min_frame_index += 1
            self.buffers = {key:buffer for key, buffer in self.buffers.items() if len(buffer) > 0}
        super().close()

    def handle_event(self, ev:TrackEvent) -> None:
        buffer = self.buffers[ev.track_id]
        buffer.append(ev)
        
        if len(buffer) > self.buffer_size:
            current_index = buffer[0].frame_index
            for index in range(self.min_frame_index, current_index+1):
                self.publish_events_at_frame_index(index)
            self.min_frame_index = current_index+1

    def publish_events_at_frame_index(self, upto_index:int) -> int:
        for buffer in self.buffers.values():
            while len(buffer) > 0 and buffer[0].frame_index <= upto_index:
                self.publish_event(buffer.pop(0))