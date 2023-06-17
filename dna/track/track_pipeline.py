from __future__ import annotations

from typing import Optional
from omegaconf import OmegaConf
import numpy as np

from dna import color, BGR, Image, Frame
from dna.camera import FrameProcessor
from .types import ObjectTrack, ObjectTracker, TrackProcessor
from .track_processors import TrailCollector, TrackCsvWriter


class TrackingPipeline(FrameProcessor):
    __slots__ = ( 'tracker', '_trail_collector', '_track_processors', '_draw')

    def __init__(self, tracker:ObjectTracker, draw:list[str]=[]) -> None:
        """TrackingPipeline을 생성한다.

        Args:
            tracker (ObjectTracker): TrackEvent를 생성할 tracker 객체.
            draw (list[str], optional):Tracking 과정에서 영상에 표시할 항목 리스트.
            리스트에는 'track_zones', 'exit_zones', 'stable_zones', 'magnifying_zones', 'tracks'이 포함될 수 있음.
            Defaults to [].
        """
        super().__init__()

        self.tracker = tracker
        self._trail_collector = TrailCollector()
        self._track_processors:list[TrackProcessor] = [self._trail_collector]
        self._draw = draw

    @staticmethod
    def load(tracker_conf:OmegaConf) -> TrackingPipeline:
        from .dna_tracker import DNATracker
        tracker = DNATracker.load(tracker_conf)
        
        draw = tracker_conf.get("draw", [])
        tracking_pipeline = TrackingPipeline(tracker=tracker, draw=draw)

        if output := tracker_conf.get("output", None):
            tracking_pipeline.add_track_processor(TrackCsvWriter(output))
            
        return tracking_pipeline
        
    def add_track_processor(self, proc:TrackProcessor) -> None:
        self._track_processors.append(proc)

    def on_started(self, capture) -> None:
        for processor in self._track_processors:
            processor.track_started(self.tracker)

    def on_stopped(self) -> None:
        for processor in self._track_processors:
            processor.track_stopped(self.tracker)

    def set_control(self, key:int) -> int:
        def toggle(tag:str):
            if tag in self._draw:
                self._draw.pop(tag)
            else:
                self._draw.append(tag)
            
        if key == ord('t'):
            toggle('tracks')
        if key == ord('b'):
            toggle('blind_zones')
        if key == ord('z'):
            toggle('track_zones')
        if key == ord('e'):
            toggle('exit_zones')
        if key == ord('s'):
            toggle('stable_zones')
        if key == ord('m'):
            toggle('magnifying_zones')
        return key

    def process_frame(self, frame:Frame) -> Frame:
        tracks = self.tracker.track(frame)

        for processor in self._track_processors:
            processor.process_tracks(self.tracker, frame, tracks)

        if self._draw:
            convas = frame.image
            if 'track_zones' in self._draw:
                for zone in self.tracker.params.track_zones:
                    convas = zone.draw(convas, color.RED, 1)
            if 'blind_zones' in self._draw:
                for zone in self.tracker.params.blind_zones:
                    convas = zone.draw(convas, color.YELLOW, 1)
            if 'exit_zones' in self._draw:
                for zone in self.tracker.params.exit_zones:
                    convas = zone.draw(convas, color.RED, 1)
            if 'stable_zones' in self._draw:
                for zone in self.tracker.params.stable_zones:
                    convas = zone.draw(convas, color.BLUE, 1)
            if 'magnifying_zones' in self._draw:
                for roi in self.tracker.params.magnifying_zones:
                    roi.draw(convas, color.ORANGE, line_thickness=1)

            if 'tracks' in self._draw:
                tracks = self.tracker.tracks
                for track in tracks:
                    if hasattr(track, 'last_detection'):
                        det = track.last_detection
                        if det:
                            convas = det.draw(convas, color.WHITE, line_thickness=1)
                for track in tracks:
                    if track.is_tentative():
                        convas = self.draw_track_trail(convas, track, color.RED, trail_color=color.BLUE, line_thickness=1)
                for track in sorted(tracks, key=lambda t:t.id, reverse=True):
                    if track.is_confirmed():
                        convas = self.draw_track_trail(convas, track, color.BLUE, trail_color=color.RED, line_thickness=1)
                    elif track.is_temporarily_lost():
                        convas = self.draw_track_trail(convas, track, color.BLUE, trail_color=color.LIGHT_GREY, line_thickness=1)
            return Frame(convas, frame.index, frame.ts)
        else:
            return frame
    
    def draw_track_trail(self, convas:Image, track:ObjectTrack, color:color.BGR, label_color:BGR=color.WHITE,
                        trail_color:Optional[BGR]=None, line_thickness=2) -> np.ndarray:
        convas = track.draw(convas, color, label_color=label_color, line_thickness=line_thickness)

        if trail_color:
            trail = self._trail_collector.get_trail(track.id)
            return trail.draw(convas, trail_color, line_thickness=line_thickness)
