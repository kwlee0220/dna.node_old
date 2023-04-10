from __future__ import annotations
from typing import Tuple, Dict, Optional, Any

import logging

from ..types import TrackEvent
from ..event_processor import EventProcessor
from .types import TrackDeleted, LineTrack


class ToLineTransform(EventProcessor):
    __slots__ = ( 'last_events', 'logger' )
    
    def __init__(self, logger:logging.Logger) -> None:
        EventProcessor.__init__(self)
        self.last_events:Dict[str,TrackEvent] = dict()
        self.logger = logger

    def close(self) -> None:
        self.last_events.clear()
        super().close()

    def handle_event(self, ev:Any) -> None:
        if isinstance(ev, TrackEvent):
            self.handle_track_event(ev)
        else:
            self._publish_event(ev)

    def handle_track_event(self, ev:TrackEvent) -> None:
        if not ev.is_deleted():
            # track의 첫번재 이벤트인 경우는 last_event가 ev(자기 자신)이 됨.
            last_event = self.last_events.get(ev.track_id, ev)
            self._publish_event(LineTrack.from_events(last_event, ev))
            self.last_events[ev.track_id] = ev
        else:
            self.last_events.pop(ev.track_id, None)
            self._publish_event(TrackDeleted(track_id=ev.track_id, frame_index=ev.frame_index, ts=ev.ts, source=ev))