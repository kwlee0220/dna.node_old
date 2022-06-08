from __future__ import annotations
from typing import List
from abc import ABCMeta, abstractmethod
from pathlib import Path

from dna.node.track_event import TrackEvent


class EventListener(metaclass=ABCMeta):
    @abstractmethod
    def handle_event(self, ev: object) -> None:
        pass

    def listen(self, queue: EventQueue) -> None:
        queue.add_listener(self)

    @abstractmethod
    def close(self) -> None:
        pass

class EventQueue:
    def __init__(self) -> None:
        self.listeners:List[EventListener] = []

    def add_listener(self, listener:EventListener) -> None:
        self.listeners.append(listener)

    def close(self) -> None:
        for sub in self.listeners:
            sub.close()

    def publish_event(self, ev:object) -> None:
        for sub in self.listeners:
            sub.handle_event(ev)

class EventProcessor(EventListener, EventQueue):
    def __init__(self) -> None:
        EventListener.__init__(self)
        EventQueue.__init__(self)

    def close(self) -> None:
        EventQueue.close(self)
        EventListener.close(self)


from dna import Frame
from dna.track import Track, TrackerCallback
from .track_event import TrackEvent, EOT
class TrackEventSource(TrackerCallback, EventQueue):
    def __init__(self, node_id:str) -> None:
        TrackerCallback.__init__()
        EventQueue.__init__()

        self.node_id = node_id

    def track_started(self, tracker) -> None: pass
    def track_stopped(self, tracker) -> None:
        self.publisher.publish(EOT)

    def tracked(self, tracker, frame: Frame, tracks: List[Track]) -> None:
        for track in tracks:
            self.publish_event(TrackEvent.from_track(self.node_id, track))

import sys
class PrintTrackEvent(EventListener):
    def __init__(self, file: str) -> None:
        super().__init__()

        self.file = file

        Path(self.file).parent.mkdir(exist_ok=True)
        self.fp = open(self.file, 'w')

    def close(self) -> None:
        self.fp.close()
        self.fp = None

    def handle_event(self, ev: object) -> None:
        self.fp.write(ev.to_json() + '\n')