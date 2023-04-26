from __future__ import annotations
import sys
from typing import List, Dict, Any, Union, Optional, Tuple, Callable

import logging
from collections import defaultdict
from pathlib import Path

from .types import TrackEvent, TimeElapsed
from .event_processor import EventProcessor, EventListener, EventQueue
from .refine_track_event import RefineTrackEvent


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
    def __init__(self, min_frame_index:Callable[[],int], *, logger:Optional[logging.Logger]=None) -> None:
        EventProcessor.__init__(self)

        self.groups:Dict[int,List[TrackEvent]] = defaultdict(list)
        self.min_frame_index = min_frame_index
        self.max_published_index = 0
        self.logger = logger
    
    def close(self) -> None:
        while self.groups:
            min_frame_index = min(self.groups.keys())
            group = self.groups.pop(min_frame_index, None)
            self._publish_event(group)
        super().close()

    def handle_event(self, ev:TrackEvent|TimeElapsed) -> None:
        if isinstance(ev, TrackEvent):
            # 만일 새 TrackEvent가 이미 publish된 track event group의 frame index보다 작은 경우
            # late-arrived event 문제가 발생하여 예외를 발생시킨다.
            if ev.frame_index <= self.max_published_index:
                raise ValueError(f'late arrived TrackEvent: {ev}')
            
            group = self.groups[ev.frame_index]
            group.append(ev)
            
            # pending된 TrackEvent group 중에서 가장 작은 frame index를 갖는 group을 검색.
            frame_index, group = min(self.groups.items(), key=lambda t: t[0])
            
            # 본 GroupByFrameIndex 이전 EventProcessor들에서 pending된 TrackEvent 들 중에서
            # 가장 작은 frame index를 알아내어, 이 frame index보다 작은 값을 갖는 group의 경우에는
            # 이후 해당 group에 속하는 TrackEvent가 도착하지 않을 것이기 때문에 그 group들을 publish한다.
            min_frame_index = self.min_frame_index()
            if not min_frame_index:
                min_frame_index = ev.frame_index
            
            for idx in range(frame_index, min_frame_index):
                group = self.groups.pop(idx, None)
                if group:
                    if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f'publish TrackEvent group: frame_index={idx}, count={len(group)}')
                    self._publish_event(group)
                    self.max_published_index = max(self.max_published_index, idx)

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