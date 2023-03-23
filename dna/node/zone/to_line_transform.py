from __future__ import annotations
from typing import Tuple, Dict, Optional, Any

import logging

from dna.tracker import TrackState
from ..types import TrackEvent
from ..event_processor import EventProcessor
from .types import LineTrack, TrackDeleted


class ToLineTransform(EventProcessor):
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
        if ev.state != TrackState.Deleted:
            last_event = self.last_events.get(ev.track_id)
            if last_event:
                self._publish_event(LineTrack.from_events(last_event, ev))
            else:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'track created: id={ev.track_id}')
            self.last_events[ev.track_id] = ev
        else:
            self.last_events.pop(ev.track_id, None)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'track deleted: id={ev.track_id}')
            self._publish_event(TrackDeleted(track_id=ev.track_id, frame_index=ev.frame_index, ts=ev.ts))