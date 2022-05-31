from typing import List
from abc import ABCMeta, abstractmethod
from pathlib import Path

from dna.node.track_event import TrackEvent


class EventSubscriber(metaclass=ABCMeta):
    @abstractmethod
    def handle_event(self, ev: object) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

class EventQueue:
    def __init__(self) -> None:
        self.subscribers:List[EventSubscriber] = []

    def subscribe(self, sub:EventSubscriber) -> None:
        self.subscribers.append(sub)

    def close(self) -> None:
        for sub in self.subscribers:
            sub.close()

    def publish_event(self, ev:object) -> None:
        for sub in self.subscribers:
            sub.handle_event(ev)


class EventProcessor(EventSubscriber):
    def __init__(self) -> None:
        EventSubscriber.__init__(self)
        
        self.queue = EventQueue()

    def subscribe(self, sub:EventSubscriber) -> None:
        self.queue.subscribe(sub)

    def publish_event(self, ev:object) -> None:
        self.queue.publish_event(ev)

    def close(self) -> None:
        self.queue.close()


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
class PrintTrackEvent(EventSubscriber):
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