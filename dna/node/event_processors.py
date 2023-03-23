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


def order_by_frame_index(input_queue:EventQueue, max_pending_frames:int, timeout:float) -> EventQueue:
    groupby = GroupByFrameIndex(max_pending_frames=max_pending_frames, timeout=timeout)
    input_queue.add_listener(groupby)

    ungroup = UngroupEvent()
    groupby.add_listener(ungroup)

    return ungroup


class UngroupEvent(EventProcessor):
    def handle_event(self, group) -> None:
        if isinstance(group, list):
            for ev in group:
                self._publish_event(ev)
        else:
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


import numpy as np
from dna import Frame
from dna.camera import FrameProcessor, ImageProcessor
from dna.tracker import TrackState
from dna.tracker.feature_extractor import FeatureExtractor
class CollectReIDFeatures(FrameProcessor,EventProcessor):
    def __init__(self, extractor:FeatureExtractor, frame_buffer_size:int=5) -> None:
        super().__init__()
        
        self.extractor = extractor
        self.buffer_size = frame_buffer_size
        self.groupby = GroupByFrameIndex(frame_buffer_size, timeout=5.0)
        self.frame_buffer:List[Frame] = []
    
    def handle_event(self, group:List[TrackEvent]) -> None:
        index = group[0].frame_index.frame_index
        if index > self.frame_buffer[-1].index:
            self.event_buffer.clear()
        elif index >= self.frame_buffer[0].index:
            first = group[0]
            offset = index - self.frame_buffer[0].index
            frame = self.frame_buffer[offset]
            self.frame_buffer = self.frame_buffer[offset+1:]
            
            tracks = [track for track in group if track.is_confirmed() or track.is_tentative()]
            boxes = [track.location for track in tracks]
            for track, feature in zip(tracks, self.extractor.extract_boxes(frame.image, boxes)):
                feature_binary = np.binary_repr(feature)
                (track.node_id, track.track_id, feature_binary, track.frame_index)
            
    
    def on_started(self, proc:ImageProcessor) -> None: pass
    def on_stopped(self) -> None: pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        self.frame_buffer.append(frame)
        self.frame_buffer = self.frame_buffer[-self.buffer_size:]
        return frame


# class TaggedEventProcessor(EventProcessor):
#     def __init__(self, tag:Any) -> None:
#         EventProcessor.__init__(self)
#         self.tag = tag
        
#     def handle_event(self, ev:Any) -> None:
#         self.publish_event((self.tag, ev))

# class JoinByFrameIndex(EventProcessor):
#     def __init__(self, input_queues:List[EventQueue], buffer_size:int) -> None:
#         EventProcessor.__init__(self)

#         self.buffer_size = buffer_size
#         self.event_buffers:Dict[int,List[Any]] = dict()
#         self.input_queues:List[TaggedEventProcessor] = []
#         for idx, queue in enumerate(input_queues):
#             tagged_listener = TaggedEventProcessor(idx)
#             tagged_listener.add_listener(self)
#             queue.add_listener(tagged_listener)
#             self.input_queues.append(tagged_listener)
#             self.event_buffers[idx] = []
    
#     def close(self) -> None:
#         for queue in self.input_queues:
#             queue.close()
#         super().close()

#     def handle_event(self, tagged_event:Tuple[Any,Any]) -> None:
#         tag, ev = tagged_event
#         buffer = self.event_buffers[tag]
#         buffer.append(ev)

#         buffers = list(self.event_buffers.values())
#         first_index = buffers[0][0].frame_index if buffers[0] else None
#         if not first_index:
#             self.flush_overflow_buffer()
#             return

#         for buffer in buffers[1:]:
#             index = buffer[0].frame_index if buffer else None
#             if first_index != index:
#                 self.flush_overflow_buffer()
#                 return

#         joined = [buffer.pop(0) for buffer in self.event_buffers.values()]
#         self.publish_event(joined)

#     def flush_overflow_buffer(self) -> None:
#         overflows = [event_buffer for event_buffer in self.event_buffers.values() if len(event_buffer) > self.buffer_size]
#         if overflows:
#             target_index = overflows[0][0].frame_index
#             min_index = self._find_min_frame_index()
#             for index in range(min_index, target_index+1):
#                 self._flush_event_at_frame(index)

#     def _find_min_frame_index(self) -> None:
#         return min(event_buffer[0].frame_index if event_buffer else sys.maxsize for event_buffer in self.event_buffers.values())

#     def _flush_event_at_frame(self, frame_index:int) -> None:
#         join = [None] * len(self.event_buffers)
#         for idx, event_buffer in enumerate(self.event_buffers.values()):
#             first_index = event_buffer[0].frame_index if event_buffer else -1
#             if first_index == frame_index:
#                 join[idx] = event_buffer.pop(0)

#         self.publish_event(join)
