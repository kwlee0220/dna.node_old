from typing import Optional, Any
from abc import ABCMeta, abstractmethod
from pathlib import Path

from pubsub import PubSub, Queue
from dna.node.track_event import TrackEvent
from .utils import EventPublisher


class EventProcessor(metaclass=ABCMeta):
    def __init__(self, in_queue: Queue, publisher: Optional[EventPublisher]=None) -> None:
        self.in_queue = in_queue
        self.publisher = publisher

    def subscribe(self) -> Queue:
        if self.publisher:
            return self.publisher.subscribe()
        else:
            raise AssertionError(f"Processor cannot publish events: {self}")

    @abstractmethod
    def handle_event(self, ev: TrackEvent) -> None:
        pass

    def close(self) -> None:
        pass

    def publish_event(self, ev) -> None:
        if self.publisher:
            self.publisher.publish(ev)
        else:
            raise AssertionError(f"Processor cannot publish events: {self}")

    def run(self) -> None:
        for entry in self.in_queue.listen():
            event = entry['data']
            if event.luid is None:
                break
            self.handle_event(event)

        if self.publisher:
            self.publish_event(event)
        self.close()


import sys
class PrintTrackEvent(EventProcessor):
    def __init__(self, queue, file: str) -> None:
        super().__init__(queue)

        self.file = file
        self.fp = sys.stdout if self.file == '-' else open(self.file, 'w')

    def close(self) -> None:
        if self.file != '-':
            self.fp.close()
        self.fp = None

    def subscribe(self): pass

    def handle_event(self, ev: TrackEvent) -> None:
        self.fp.write(ev.to_csv() + '\n')