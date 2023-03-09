from __future__ import annotations

from typing import List, Dict, Set
from collections import defaultdict

from dna.tracker import TrackState
from .track_event import TrackEvent
from .event_processor import EventProcessor

import logging
LOGGER = logging.getLogger('dna.node.event')


class DropShortTrail(EventProcessor):
    __slots__ = 'min_trail_length', 'long_trails', 'pending_dict'

    def __init__(self, min_trail_length:int) -> None:
        EventProcessor.__init__(self)

        self.min_trail_length = min_trail_length
        self.long_trails: Set[str] = set()  # 'long trail' 여부
        self.pending_dict: Dict[str, List[TrackEvent]] = defaultdict(list)

    def close(self) -> None:
        super().close()
        self.pending_dict.clear()
        self.long_trails.clear()

    def handle_event(self, ev) -> None:
        is_long_trail = ev.track_id in self.long_trails
        if ev.state == TrackState.Deleted:   # tracking이 종료된 경우
            if is_long_trail:
                self.long_trails.discard(ev.track_id)
                self.publish_event(ev)
            else:
                pendings = self.pending_dict.pop(ev.track_id, [])
                if pendings:
                    LOGGER.info(f"drop short track events: track_id={ev.track_id}, length={len(pendings)}")
        elif is_long_trail:
            self.publish_event(ev)
        else:
            pendings = self.pending_dict[ev.track_id]
            pendings.append(ev)

            # pending된 event의 수가 threshold (min_trail_length) 이상이면 long-trail으로 설정하고,
            # 더 이상 pending하지 않고, 바로 publish 시킨다.
            if len(pendings) >= self.min_trail_length:
                self.pending_dict.pop(ev.track_id, None)
                self.__publish_pendings(pendings)
                self.long_trails.add(ev.track_id)

    def __publish_pendings(self, pendings:List[TrackEvent]) -> None:
        for pev in pendings:
            self.publish_event(pev)