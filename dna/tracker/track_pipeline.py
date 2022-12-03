from __future__ import annotations
from typing import List, Optional
import collections
from pathlib import Path

from omegaconf import OmegaConf
import numpy as np

from dna import plot_utils, color, Point, BGR, Image, Frame
from .tracker import Track, TrackState, ObjectTracker, TrackProcessor, DetectionBasedObjectTracker


class TrackWriter(TrackProcessor):
    def __init__(self, track_file: str) -> None:
        super().__init__()

        self.track_file = track_file
        self.out_handle = None

    def track_started(self, tracker: ObjectTracker) -> None:
        super().track_started(tracker)

        parent = Path(self.track_file).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self.out_handle = open(self.track_file, 'w')
    
    def track_stopped(self, tracker: ObjectTracker) -> None:
        if self.out_handle:
            self.out_handle.close()
            self.out_handle = None

        super().track_stopped(tracker)

    def process_tracks(self, tracker: ObjectTracker, frame: Frame, tracks: List[Track]) -> None:
        for track in tracks:
            self.out_handle.write(track.to_string() + '\n')

class Trail:
    __slots__ = ('__tracks', )

    def __init__(self) -> None:
        self.__tracks = []

    @property
    def tracks(self) -> List[Track]:
        return self.__tracks

    def append(self, track: Track) -> None:
        self.__tracks.append(track)

    def draw(self, convas: np.ndarray, color: color.BGR, line_thickness=2) -> np.ndarray:
        # track의 중점 값들을 선으로 이어서 출력함
        track_centers: List[Point] = [t.location.center() for t in self.tracks[-11:]]
        return plot_utils.draw_line_string(convas, track_centers, color, line_thickness)
    

class TrailCollector(TrackProcessor):
    __slots__ = ('trails', )

    def __init__(self) -> None:
        super().__init__()
        self.trails = collections.defaultdict(lambda: Trail())

    def get_trail(self, track_id: str) -> Trail:
        return self.trails[track_id]

    def track_started(self, tracker: ObjectTracker) -> None: pass
    def track_stopped(self, tracker: ObjectTracker) -> None: pass

    def process_tracks(self, tracker: ObjectTracker, frame: Frame, tracks: List[Track]) -> None:      
        for track in tracks:
            if track.state == TrackState.Confirmed  \
                or track.state == TrackState.TemporarilyLost    \
                or track.state == TrackState.Tentative:
                self.trails[track.id].append(track)
            elif track.state == TrackState.Deleted:
                self.trails.pop(track.id, None)


from dna.camera import ImageProcessor, FrameProcessor
class TrackingPipeline(FrameProcessor):
    __slots__ = ( 'tracker', 'is_detection_based', 'trail_collector', 'track_processors', 'draw_tracks', 'show_zones' )

    @classmethod
    def load(cls, img_proc: ImageProcessor, tracker_conf: OmegaConf,
                            track_processors: List[TrackProcessor]=[]) -> TrackingPipeline:
        tracker_uri = tracker_conf.get("uri", "dna.tracker.dna_deepsort")
        parts = tracker_uri.split(':', 1)
        id, query = tuple(parts) if len(parts) > 1 else (tracker_uri, "")

        from dna import Box
        domain = Box.from_size(img_proc.capture.size)
        
        import importlib
        tracker_module = importlib.import_module(id)
        tracker = tracker_module.load(domain, tracker_conf)

        draw_tracks = img_proc.is_drawing() and tracker_conf.get("draw_tracks", True)
        draw_zones = img_proc.is_drawing() and tracker_conf.get("draw_zones", False)

        output = tracker_conf.get("output", None)
        if output is not None:
            track_processors = [TrackWriter(output)] + track_processors
            
        return cls(tracker=tracker, processors=track_processors, draw_tracks=draw_tracks, draw_zones=draw_zones)

    def __init__(self, tracker: ObjectTracker, processors: List[TrackProcessor]=[], draw_tracks: bool=False,
                draw_zones=False) -> None:
        super().__init__()

        self.tracker = tracker
        self.is_detection_based = isinstance(self.tracker, DetectionBasedObjectTracker)
        self.trail_collector = TrailCollector()
        self.track_processors = processors + [self.trail_collector]
        self.draw_tracks = draw_tracks
        self.draw_zones = draw_zones

    def on_started(self, capture) -> None:
        for processor in self.track_processors:
            processor.track_started(self.tracker)

    def on_stopped(self) -> None:
        for processor in self.track_processors:
            processor.track_stopped(self.tracker)

    def set_control(self, key: int) -> int:
        if key == ord('z'):
            self.draw_zones = not self.draw_zones
        if key == ord('t'):
            self.draw_tracks = not self.draw_tracks
        return key

    def process_frame(self, frame: Frame) -> Frame:
        tracks = self.tracker.track(frame)

        for processor in self.track_processors:
            processor.process_tracks(self.tracker, frame, tracks)

        convas = frame.image
        if self.draw_zones:
            for zone in self.tracker.params.blind_zones:
                convas = plot_utils.draw_polygon(convas, list(zone.exterior.coords), color.YELLOW, 2)
            for zone in self.tracker.params.exit_zones:
                convas = plot_utils.draw_polygon(convas, list(zone.exterior.coords), color.RED, 2)
            for poly in self.tracker.params.stable_zones:
                convas = plot_utils.draw_polygon(convas, list(poly.exterior.coords), color.BLUE, 2)

        if self.draw_tracks:
            if self.is_detection_based:
                for det in self.tracker.last_frame_detections():
                    convas = det.draw(convas, color.WHITE, line_thickness=2)

            for track in tracks:
                if track.is_tentative():
                    convas = self.draw_track_trail(convas, track, color.RED, trail_color=color.BLUE)
            for track in sorted(tracks, key=lambda t: t.id, reverse=True):
                if not track.is_tentative():
                    if track.is_confirmed():
                        convas = self.draw_track_trail(convas, track, color.BLUE, trail_color=color.RED)
                    if track.is_temporarily_lost():
                        convas = self.draw_track_trail(convas, track, color.BLUE, trail_color=color.LIGHT_GREY)
            return Frame(convas, frame.index, frame.ts)
        else:
            return frame
    
    def draw_track_trail(self, convas:Image, track: Track, color: color.BGR, label_color: BGR=color.WHITE,
                        trail_color: Optional[BGR]=None) -> np.ndarray:
        convas = track.draw(convas, color, label_color=label_color, line_thickness=2)

        if trail_color:
            trail = self.trail_collector.get_trail(track.id)
            return trail.draw(convas, trail_color)
