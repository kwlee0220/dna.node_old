from __future__ import annotations
import sys
from typing import List, Dict
from abc import ABCMeta, abstractmethod

from collections import defaultdict
from pathlib import Path

from .track_event import TrackEvent
from .event_processor import EventProcessor

_STDOUT_STDERR = set(('stdout', 'stderr'))


class TrackEventWriter(EventProcessor):
    def __init__(self, file_path: str) -> None:
        EventProcessor.__init__(self)

        self.file_path = file_path
        if self.file_path == 'stdout':
            self.fp = sys.stdout
        elif self.file_path == 'stderr':
            self.fp = sys.stderr
        else:
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
            self.fp = open(self.file_path, 'w')
    
    def close(self) -> None:
        if self.file_path not in _STDOUT_STDERR and self.fp:
            self.fp.close()
            self.fp = None
        super().close()

    def handle_event(self, ev:TrackEvent) -> None:
        raise NotImplementedError()


class CsvTrackEventWriter(TrackEventWriter):
    def __init__(self, file_path: str) -> None:
        TrackEventWriter.__init__(self, file_path)

    def handle_event(self, ev:TrackEvent) -> None:
        x1, y1, x2, y2 = tuple(ev.location.tlbr)
        millis = int(round(ev.ts * 1000))
        line = f"{ev.frame_index},{ev.track_id},{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f},{ev.state.name},{millis}"
        self.fp.write(line + '\n')
        

class JsonTrackEventWriter(TrackEventWriter):
    def __init__(self, file_path: str) -> None:
        TrackEventWriter.__init__(self, file_path)

    def handle_event(self, ev: object) -> None:
        self.fp.write(ev.to_json() + '\n')
