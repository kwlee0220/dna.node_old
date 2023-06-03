from __future__ import annotations

from typing import List, Optional, Dict, Tuple

import numpy as np

from dna import Frame, utils, Size2d
from dna.camera import FrameProcessor, ImageProcessor
from .types import TrackEvent, TrackFeature, TrackId
from .event_processor import EventProcessor
from dna.tracker.feature_extractor import DeepSORTMetricExtractor


class PublishReIDFeatures(FrameProcessor,EventProcessor):
    MAX_FRAME_BUFFER_LENGTH = 80
    
    def __init__(self, extractor:DeepSORTMetricExtractor,
                 distinct_distance:float=1,
                 min_crop_size:Size2d=Size2d([90,90])) -> None:
        super().__init__()
        
        self.extractor = extractor
        self.distinct_distance = distinct_distance
        self.min_crop_size = min_crop_size
        self.frame_buffer:List[Frame] = []
        self.pending_reid_tracks:List[Tuple[int, List[TrackEvent]]] = []
        self.representives:Dict[TrackId,np.ndarray] = dict()
    
    def handle_event(self, group:List[TrackEvent]) -> None:
        # frame_buffer에 해당 frame 존재 여부와 무관하게 'delete'된 track 처리를 수행함.
        reid_tracks = []
        for track in group:
            if track.is_confirmed() or track.is_tentative():
                # ReID feature의 안정성 향상을 위해 detection box의 크기가
                # 일정 크기 이상인 것만 reid feature를 추출한다.
                if track.detection_box.size() >= self.min_crop_size:
                    reid_tracks.append(track)
            elif track.is_deleted():
                self.representives.pop(track.track_id, None)
                # track이 종료됨을 알리기 위해 feature값이 None인 'TrackFeature' 객체를 publish한다.
                self._publish_event(TrackFeature(node_id=track.node_id, track_id=track.track_id, feature=None,
                                                 zone_relation=track.zone_relation, ts=track.ts))
        
        frame_index = group[0].frame_index
        self.pending_reid_tracks.append((frame_index, reid_tracks))
        
        while self.pending_reid_tracks:
            frame_index, reid_tracks = self.pending_reid_tracks[0]
            frame = self.pop_frame(frame_index)
            if not frame:
                break
            
            self.pending_reid_tracks.pop(0)
            if reid_tracks:
                self._publish_reid_features(reid_tracks, frame)
              
    def on_started(self, proc:ImageProcessor) -> None: pass
    def on_stopped(self) -> None: pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > PublishReIDFeatures.MAX_FRAME_BUFFER_LENGTH:
            self.frame_buffer.pop(0)
        return frame
            
    def _publish_reid_features(self, reid_tracks:List[TrackEvent], frame:Frame) -> None:
        def to_feature(track:TrackEvent, feature:np.ndarray) -> TrackFeature:
            return TrackFeature(node_id=track.node_id, track_id=track.track_id, feature=feature,
                                zone_relation=track.zone_relation, ts=track.ts)
            
        boxes = [track.detection_box for track in reid_tracks]
        for track, feature in zip(reid_tracks, self.extractor.extract_boxes(frame.image, boxes)):
            reprenative = self.representives.get(track.track_id)
            if reprenative is not None:
                dist = self.extractor.distance(reprenative, feature)
                if dist < self.distinct_distance:
                    continue
            self.representives[track.track_id] = feature
            self._publish_event(to_feature(track, feature))
        
    def pop_frame(self, frame_index:int) -> Optional[Frame]:
        for idx, pending_frame in enumerate(self.frame_buffer):
            if frame_index == pending_frame.index:
                self.frame_buffer = self.frame_buffer[idx+1:]
                return pending_frame
            elif frame_index < pending_frame.index:
                if idx == 0:
                    return None
                    # raise AssertionError(f"ReID feature extraction failed: target frame has been purged already: frame={frame_index}, {self}")
                break
        return None
        
    def __repr__(self) -> str:
        frames_str = f', frame_buffer[{self.frame_buffer[0].index}:{self.frame_buffer[-1].index}]' if self.frame_buffer else ""
        return f'{self.__class__.__name__}[min_det_size={self.min_crop_size}{frames_str}, tracks={list(self.representives.keys())}]'