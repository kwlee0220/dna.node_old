from __future__ import annotations
from typing import Tuple, List, Dict, Set, Optional, Any

import logging
from dataclasses import dataclass, field
import itertools

from ..event_processor import EventQueue, EventListener, EventProcessor
from .zone_sequence_collector import ZoneSequence


@dataclass(frozen=True)
class Motion:
    track_id: int
    id: str
    frame_index: int
    ts: float

    def __repr__(self) -> str:
        return f'Motion[track={self.track_id}, motion={self.id}, frame={self.frame_index}]'
    

class MotionDetector(EventProcessor):
    def __init__(self, motion_definitons:Dict[str,str], logger:logging.Logger) -> None:
        super().__init__() 
        self.motion_definitions = motion_definitons
        self.logger = logger

    def handle_event(self, ev:ZoneSequence) -> None:
        if isinstance(ev, ZoneSequence):
            seq = ''.join([zone.zone_id for zone in ev])
            seq = ''.join(i for i, _ in itertools.groupby(seq))
            motion_id = self.motion_definitions.get(seq)
            if motion_id:
                motion = Motion(track_id=ev.track_id, id=motion_id, frame_index=ev.frame_index, ts=ev.ts)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'detect motion: track={ev.track_id}, seq={seq}, motion={motion.id}, frame={ev.frame_index}')
                self.publish_event(motion)
            else:
                if self.logger.isEnabledFor(logging.WARN):
                    self.logger.warn(f'unknown motion: track={ev.track_id}, seq={seq}, frame={ev.frame_index}')
    