from __future__ import annotations
from typing import Tuple, Dict, Optional

import logging

from dna.tracker import TrackState
from ..track_event import TrackEvent
from ..event_processor import EventProcessor
from .types import LineTrack, TrackDeleted


class ToLineTransform(EventProcessor):
    def __init__(self, logger:logging.Logger) -> None:
        EventProcessor.__init__(self)
        self.last_events:Dict[int,TrackEvent] = dict()
        self.logger = logger

    def close(self) -> None:
        self.last_events.clear()
        super().close()

    def handle_event(self, ev:TrackEvent) -> None:
        last_event = self.last_events.get(ev.track_id)
        if last_event:
            self.publish_event(LineTrack.from_events(last_event, ev))
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'track created: id={ev.track_id}')
        self.last_events[ev.track_id] = ev

        if ev.state == TrackState.Deleted:
            self.last_events.pop(ev.track_id, None)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'track deleted: id={ev.track_id}')
            self.publish_event(TrackDeleted(track_id=ev.track_id, frame_index=ev.frame_index, ts=ev.ts))



        # if ev.state == TrackState.Deleted:
        #     self.last_events.pop(ev.track_id, None)
        #     if self.logger.isEnabledFor(logging.DEBUG):
        #         self.logger.debug(f'track deleted: id={ev.track_id}')
        #     self.publish_event(TrackDeleted(track_id=ev.track_id, frame_index=ev.frame_index, ts=ev.ts))
        # else:
        #     last_event = self.last_events.get(ev.track_id)
        #     if last_event:
        #         self.publish_event(LineTrack.from_events(last_event, ev))
        #     else:
        #         if self.logger.isEnabledFor(logging.DEBUG):
        #             self.logger.debug(f'track created: id={ev.track_id}')
        #     self.last_events[ev.track_id] = ev