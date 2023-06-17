from __future__ import annotations

import collections
from pathlib import Path


from dna import Frame, color, Point
from .track_state import TrackState
from dna.support import plot_utils
from .types import ObjectTrack, ObjectTracker, TrackProcessor


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

    def process_tracks(self, tracker:ObjectTracker, frame:Frame, tracks:list[ObjectTrack]) -> None:
        for track in tracks:
            self.out_handle.write(track.to_csv() + '\n')
            

class Trail:
    __slots__ = ('__tracks', )

    def __init__(self) -> None:
        self.__tracks = []

    @property
    def tracks(self) -> list[ObjectTrack]:
        return self.__tracks

    def append(self, track:ObjectTrack) -> None:
        self.__tracks.append(track)

    def draw(self, convas:np.ndarray, color:color.BGR, line_thickness=2) -> np.ndarray:
        # track의 중점 값들을 선으로 이어서 출력함
        track_centers:list[Point] = [t.location.center() for t in self.tracks[-11:]]
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

    def process_tracks(self, tracker:ObjectTracker, frame:Frame, tracks:list[ObjectTrack]) -> None:      
        for track in tracks:
            if track.state == TrackState.Confirmed  \
                or track.state == TrackState.TemporarilyLost    \
                or track.state == TrackState.Tentative:
                self.trails[track.id].append(track)
            elif track.state == TrackState.Deleted:
                self.trails.pop(track.id, None)