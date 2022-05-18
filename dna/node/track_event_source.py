from __future__ import annotations

from typing import List

from pubsub import PubSub, Queue

from dna import Frame
from dna.track import Track, TrackerCallback
from .track_event import TrackEvent, EOT
from .utils import EventPublisher


class TrackEventSource(TrackerCallback):
    __slots__ = ('publisher',)

    def __init__(self, node_id:str, publisher:EventPublisher) -> None:
        super().__init__()

        self.node_id = node_id
        self.publisher = publisher

    def subscribe(self) -> Queue:
        return self.publisher.subscribe()

    def track_started(self, tracker) -> None: pass
    def track_stopped(self, tracker) -> None:
        self.publisher.publish(EOT)

    def tracked(self, tracker, frame: Frame, tracks: List[Track]) -> None:
        for track in tracks:
            self.publisher.publish(TrackEvent.from_track(self.node_id, track))