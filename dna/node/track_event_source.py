from __future__ import annotations

from typing import List

from dna import Frame
from dna.tracker import Track, TrackerCallback
from .track_event import TrackEvent
from .event_processor import EventQueue


class TrackEventSource(TrackerCallback, EventQueue):
    def __init__(self, node_id:str) -> None:
        TrackerCallback.__init__(self)
        EventQueue.__init__(self)

        self.node_id = node_id

    def track_started(self, tracker) -> None: pass
    def track_stopped(self, tracker) -> None:
        self.close()

    def tracked(self, tracker, frame: Frame, tracks: List[Track]) -> None:
        for track in tracks:
            self.publish_event(TrackEvent.from_track(self.node_id, track))