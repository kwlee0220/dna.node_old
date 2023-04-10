from __future__ import annotations

from typing import List, Optional,Dict

import numpy as np

from dna import Frame, utils, Size2d
from dna.camera import FrameProcessor, ImageProcessor
from .types import TrackEvent, TrackFeature, TrackId
from .event_processor import EventProcessor
from dna.tracker.feature_extractor import DeepSORTMetricExtractor


class PublishReIDFeatures(FrameProcessor,EventProcessor):
    def __init__(self, frame_buffer_size:int,
                 extractor:DeepSORTMetricExtractor,
                 distinct_distance:float=1,
                 min_crop_size:Size2d=Size2d([90,90])) -> None:
        super().__init__()
        
        self.extractor = extractor
        self.buffer_size = frame_buffer_size
        self.distinct_distance = distinct_distance
        self.min_crop_size = min_crop_size
        self.frame_buffer:List[Frame] = []
        self.representives:Dict[TrackId,np.ndarray] = dict()
    
    def handle_event(self, group:List[TrackEvent]) -> None:
        def to_feature(track:TrackEvent, feature:np.ndarray) -> TrackFeature:
            return TrackFeature(node_id=track.node_id, track_id=track.track_id, feature=feature,
                                zone_relation=track.zone_relation, ts=track.ts)
        
        frame_index = group[0].frame_index
        
        # frame_buffer에 해당 frame 존재 여부와 무관하게 'delete'된 track 처리를 수행함.
        tracks = []
        for track in group:
            if track.is_confirmed() or track.is_tentative():
                if track.detection_box.size() >= self.min_crop_size:
                    tracks.append(track)
            elif track.is_deleted():
                self.representives.pop(track.track_id, None)
                self._publish_event(TrackFeature(node_id=track.node_id, track_id=track.track_id, feature=None,
                                                 zone_relation=track.zone_relation, ts=track.ts))
        
        if frame_index > self.frame_buffer[-1].index:
            self.frame_buffer.clear()
        elif frame_index >= self.frame_buffer[0].index:
            offset = frame_index - self.frame_buffer[0].index
            frame = self.frame_buffer[offset]
            self.frame_buffer = self.frame_buffer[offset+1:]
            
            if tracks:
                boxes = [track.detection_box for track in tracks]
                for track, feature in zip(tracks, self.extractor.extract_boxes(frame.image, boxes)):
                    reprenative = self.representives.get(track.track_id)
                    if reprenative is not None:
                        dist = self.extractor.distance(reprenative, feature)
                        if dist < self.distinct_distance:
                            continue
                    self.representives[track.track_id] = feature
                    self._publish_event(to_feature(track, feature))
              
    def on_started(self, proc:ImageProcessor) -> None: pass
    def on_stopped(self) -> None: pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        self.frame_buffer.append(frame)
        self.frame_buffer = self.frame_buffer[-self.buffer_size:]
        return frame