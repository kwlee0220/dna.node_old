from __future__ import annotations
import sys
from typing import List, Dict, Any, Union, Optional, Tuple

import logging
from collections import defaultdict
from pathlib import Path

from dna.node import TrackEvent, TimeElapsed, EventProcessor, EventListener, EventQueue


class PrintEvent(EventListener):
    def close(self) -> None: pass
    def handle_event(self, ev: Any) -> None:
        print(ev)           

class EventRelay(EventListener):
    def __init__(self, target:EventQueue) -> None:
        self.target = target
        
    def close(self) -> None:
        pass
    
    def handle_event(self, ev:Any) -> None:
        self.target._publish_event(ev)

class GroupByFrameIndex(EventProcessor):
    def __init__(self, max_pending_frames:int, timeout:float) -> None:
        EventProcessor.__init__(self)

        self.groups:Dict[int,List[TrackEvent]] = defaultdict(list)
        self.max_pending_frames = max_pending_frames
        self.timeout = int(round(timeout * 1000))
    
    def close(self) -> None:
        while self.groups:
            min_frame_index = min(self.groups.keys())
            group = self.groups.pop(min_frame_index, None)
            self._publish_event(group)
        super().close()

    def handle_event(self, ev:Union[TrackEvent, TimeElapsed]) -> None:
        if isinstance(ev, TrackEvent):
            group = self.groups[ev.frame_index]
            group.append(ev)
            
            if len(self.groups) > self.max_pending_frames:
                min_frame_index = min(self.groups.keys())
                group = self.groups.pop(min_frame_index, None)
                self._publish_event(group)
        else:
            self.handle_time_elapsed(ev)

    def handle_time_elapsed(self, ev:TimeElapsed) -> None:
        old_frame_indexes = [index for index, group in self.groups.items() if (ev.ts - group[0].ts) > self.timeout]
        if old_frame_indexes and self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'flush pending EventTracks at frames[{old_frame_indexes}]')
        for frame_index in old_frame_indexes:
            group = self.groups.pop(frame_index, None)
            self._publish_event(group)


class DropEventByType(EventProcessor):
    def __init__(self, event_types:List) -> None:
        super().__init__()
        self.drop_list = event_types

    def handle_event(self, ev:Union[TrackEvent,TimeElapsed]) -> None:
        if not any(ev_type for ev_type in self.drop_list if isinstance(ev, ev_type)):
            self._publish_event(ev)


class FilterEventByType(EventProcessor):
    def __init__(self, event_types:List) -> None:
        super().__init__()
        self.keep_list = event_types

    def handle_event(self, ev:Union[TrackEvent,TimeElapsed]) -> None:
        if any(ev_type for ev_type in self.keep_list if isinstance(ev, ev_type)):
            self._publish_event(ev)