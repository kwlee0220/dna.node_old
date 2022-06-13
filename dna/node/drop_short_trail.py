from __future__ import annotations

from typing import List, Dict, Set
from collections import defaultdict

from dna.tracker import TrackState
from .track_event import TrackEvent
from .event_processor import EventProcessor


class DropShortTrail(EventProcessor):
    __slots__ = 'min_track_count', 'long_trails', 'pendings'

    def __init__(self, min_trail_length:int) -> None:
        EventProcessor.__init__(self)

        self.min_trail_length = min_trail_length
        self.long_trails: Set[str] = set()
        self.pending_dict: Dict[str, List[TrackEvent]] = defaultdict(list)

    def close(self) -> None:
        super().close()
        for pendings in self.pending_dict.values():
            self.__publish_pendings(pendings)

    def handle_event(self, ev) -> None:
        if ev.state == TrackState.Deleted:   # tracking이 종료된 경우
            pendings = self.pending_dict.pop(ev.luid, [])
            if len(pendings) > self.min_trail_length:
                self.__publish_pendings(pendings)
            elif len(pendings) > 0:
                print(f"drop short track events: luid={ev.luid}, length={len(pendings)}")
                pass
            self.long_trails.discard(ev.luid)
            self.publish_event(ev)
        elif ev.luid in self.long_trails:
            self.publish_event(ev)
        else:
            pendings:List[TrackEvent] = self.pending_dict[ev.luid]
            pendings.append(ev)
            if len(pendings) >= self.min_trail_length:
                self.pending_dict.pop(ev.luid, None)
                self.__publish_pendings(pendings)
                self.long_trails.add(ev.luid)

    def __publish_pendings(self, pendings:List[TrackEvent]) -> None:
        for pev in pendings:
            self.publish_event(pev)