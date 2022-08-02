from __future__ import annotations
from typing import List
from abc import ABCMeta, abstractmethod

from pathlib import Path


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

class PrintTrackEvent(EventListener):
    def __init__(self, file: str) -> None:
        super().__init__()

        self.file = file
        Path(self.file).parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.file, 'w')

    def close(self) -> None:
        self.fp.close()
        self.fp = None

    def handle_event(self, ev: object) -> None:
        self.fp.write(ev.to_json() + '\n')