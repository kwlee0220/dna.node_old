from __future__ import annotations
from typing import Dict

import logging
import itertools

from ..event_processor import EventProcessor
from .types import Motion
from .zone_sequence_collector import ZoneSequence
    

class MotionDetector(EventProcessor):
    def __init__(self, motion_definitons:Dict[str,str], logger:logging.Logger) -> None:
        super().__init__() 
        self.motion_definitions = motion_definitons
        self.logger = logger

    def close(self) -> None:
        super().close()

    def handle_event(self, ev:ZoneSequence) -> None:
        if isinstance(ev, ZoneSequence):
            seq = ''.join([zone.zone_id for zone in ev])
            seq = ''.join(i for i, _ in itertools.groupby(seq))
            motion_id = self.motion_definitions.get(seq)
            if motion_id:
                motion = Motion(track_id=ev.track_id, id=motion_id,
                                first_frame_index=ev.first_frame_index, first_ts=ev.first_ts,
                                last_frame_index=ev.frame_index, last_ts=ev.ts)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'detect motion: track={ev.track_id}, seq={seq}, motion={motion.id}, frame={ev.frame_index}')
                self._publish_event(motion)
            else:
                if self.logger.isEnabledFor(logging.WARN):
                    self.logger.warn(f'unknown motion: track={ev.track_id}, seq={seq}, frame={ev.frame_index}')
    