from __future__ import annotations
from typing import List, Optional
from collections import defaultdict

import numpy as np

from dna import plot_utils, Point, color
from dna.camera import Frame
from .tracker import Track, TrackState, ObjectTracker, TrackerCallback


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
        
        self.trails = defaultdict(lambda: Trail())

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