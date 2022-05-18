from __future__ import annotations
from typing import List, Optional
from abc import ABCMeta, abstractmethod
from contextlib import suppress
from pathlib import Path
import collections

import numpy as np

from dna import plot_utils, color, Point, BGR, Image, Frame
from .tracker import Track, TrackState, ObjectTracker, TrackerCallback, DetectionBasedObjectTracker


class TrackWriter(TrackerCallback):
    def __init__(self, track_file: Path) -> None:
        super().__init__()

        self.track_file = track_file
        self.out_handle = None

    def track_started(self, tracker: ObjectTracker) -> None:
        super().track_started(tracker)

        self.out_handle = open(self.track_file, 'w')
    
    def track_stopped(self, tracker: ObjectTracker) -> None:
        if self.out_handle:
            self.out_handle.close()
            self.out_handle = None

        super().track_stopped(tracker)

    def tracked(self, tracker: ObjectTracker, frame: Frame, tracks: List[Track]) -> None:
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
    

class TrailCollector(TrackerCallback):
    __slots__ = ('trails', )

    def __init__(self) -> None:
        super().__init__()
        self.trails = collections.defaultdict(lambda: Trail())

    def get_trail(self, track_id: str) -> Trail:
        return self.trails[track_id]

    def track_started(self, tracker: ObjectTracker) -> None: pass
    def track_stopped(self, tracker: ObjectTracker) -> None: pass

    def tracked(self, tracker: ObjectTracker, frame: Frame, tracks: List[Track]) -> None:      
        for track in tracks:
            if track.state == TrackState.Confirmed  \
                or track.state == TrackState.TemporarilyLost    \
                or track.state == TrackState.Tentative:
                self.trails[track.id].append(track)
            elif track.state == TrackState.Deleted:
                self.trails.pop(track.id, None)



from dna.camera import ImageProcessorCallback
class ObjectTrackingCallback(ImageProcessorCallback):
    __slots__ = 'tracker', 'is_detection_based', 'trail_collector', 'callbacks', 'draw_tracks', 'show_zones'

    def __init__(self, tracker: ObjectTracker, callbacks: List[TrackerCallback]=[], draw_tracks: bool=False,
                show_zones=False) -> None:
        super().__init__()

        self.tracker = tracker
        self.is_detection_based = isinstance(self.tracker, DetectionBasedObjectTracker)
        self.trail_collector = TrailCollector()
        self.callbacks = callbacks + [self.trail_collector]
        self.draw_tracks = draw_tracks
        self.show_zones = show_zones

    def on_started(self, capture) -> None:
        for cb in self.callbacks:
            cb.track_started(self.tracker)

    def on_stopped(self) -> None:
        for cb in self.callbacks:
            cb.track_stopped(self.tracker)

    def set_control(self, key: int) -> int:
        if key == ord('r'):
            self.show_zones = not self.show_zones
        return key

    def process_image(self, frame: Frame) -> Frame:
        tracks = self.tracker.track(frame)

        for cb in self.callbacks:
            cb.tracked(self.tracker, frame, tracks)

        if self.draw_tracks:
            convas = frame.image
            if self.show_zones:
                for region in self.tracker.params.blind_zones:
                    convas = region.draw(convas, color.MAGENTA, 2)
                for region in self.tracker.params.dim_zones:
                    convas = region.draw(convas, color.RED, 2)

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