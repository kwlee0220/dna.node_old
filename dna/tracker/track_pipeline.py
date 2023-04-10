from __future__ import annotations
from typing import List, Optional
import collections
from pathlib import Path

from omegaconf import OmegaConf
import numpy as np
import cv2

from dna import plot_utils, color, Point, BGR, Image, Frame
from .types import ObjectTrack, TrackState, ObjectTracker, TrackProcessor
from . import utils


class TrackCsvWriter(TrackProcessor):
    def __init__(self, track_file:str) -> None:
        super().__init__()

        self.track_file = track_file
        self.out_handle = None

    def track_started(self, tracker:ObjectTracker) -> None:
        super().track_started(tracker)

        parent = Path(self.track_file).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self.out_handle = open(self.track_file, 'w')
    
    def track_stopped(self, tracker:ObjectTracker) -> None:
        if self.out_handle:
            self.out_handle.close()
            self.out_handle = None

        super().track_stopped(tracker)

    def process_tracks(self, tracker:ObjectTracker, frame:Frame, tracks:List[ObjectTrack]) -> None:
        for track in tracks:
            self.out_handle.write(track.to_csv() + '\n')

class Trail:
    __slots__ = ('__tracks', )

    def __init__(self) -> None:
        self.__tracks = []

    @property
    def tracks(self) -> List[ObjectTrack]:
        return self.__tracks

    def append(self, track:ObjectTrack) -> None:
        self.__tracks.append(track)

    def draw(self, convas:np.ndarray, color:color.BGR, line_thickness=2) -> np.ndarray:
        # track의 중점 값들을 선으로 이어서 출력함
        track_centers:List[Point] = [t.location.center() for t in self.tracks[-11:]]
        return plot_utils.draw_line_string(convas, track_centers, color, line_thickness)
    

class TrailCollector(TrackProcessor):
    __slots__ = ('trails', )

    def __init__(self) -> None:
        super().__init__()
        self.trails = collections.defaultdict(lambda: Trail())

    def get_trail(self, track_id:str) -> Trail:
        return self.trails[track_id]

    def track_started(self, tracker:ObjectTracker) -> None: pass
    def track_stopped(self, tracker:ObjectTracker) -> None: pass

    def process_tracks(self, tracker:ObjectTracker, frame:Frame, tracks:List[ObjectTrack]) -> None:      
        for track in tracks:
            if track.state == TrackState.Confirmed  \
                or track.state == TrackState.TemporarilyLost    \
                or track.state == TrackState.Tentative:
                self.trails[track.id].append(track)
            elif track.state == TrackState.Deleted:
                self.trails.pop(track.id, None)


from dna.camera import ImageProcessor, FrameProcessor
class TrackingPipeline(FrameProcessor):
    __slots__ = ( 'tracker', '_trail_collector', '_track_processors', 'draw')

    def __init__(self, tracker:ObjectTracker, draw:List[str]=[]) -> None:
        super().__init__()

        self.tracker = tracker
        self._trail_collector = TrailCollector()
        self._track_processors = [self._trail_collector]
        self.draw = draw

    @staticmethod
    def load(tracker_conf:OmegaConf) -> TrackingPipeline:
        tracker_uri = tracker_conf.get("uri", "dna.tracker")
        parts = tracker_uri.split(':', 1)
        id, query = tuple(parts) if len(parts) > 1 else (tracker_uri, "")
        
        import importlib
        tracker_module = importlib.import_module(id)
        tracker = tracker_module.load_dna_tracker(tracker_conf)
        
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
            if tag in self.draw:
                self.draw.pop(tag)
            else:
                self.draw.append(tag)
            
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

        if self.draw:
            convas = frame.image
            if 'track_zones' in self.draw:
                for zone in self.tracker.params.track_zones:
                    convas = zone.draw(convas, color.RED, 1)
            if 'blind_zones' in self.draw:
                for zone in self.tracker.params.blind_zones:
                    convas = zone.draw(convas, color.YELLOW, 1)
            if 'exit_zones' in self.draw:
                for zone in self.tracker.params.exit_zones:
                    convas = zone.draw(convas, color.RED, 1)
            if 'stable_zones' in self.draw:
                for zone in self.tracker.params.stable_zones:
                    convas = zone.draw(convas, color.BLUE, 1)
            if 'magnifying_zones' in self.draw:
                for roi in self.tracker.params.magnifying_zones:
                    roi.draw(convas, color.ORANGE, line_thickness=1)

            if 'tracks' in self.draw:
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
