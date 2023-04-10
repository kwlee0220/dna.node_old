from __future__ import annotations
from typing import Dict

import logging
import itertools

from dna import utils
from ..event_processor import EventProcessor
from .types import Motion, TrackDeleted
from .zone_sequence_collector import ZoneSequence
    

class MotionDetector(EventProcessor):
    def __init__(self, node_id:str, motion_definitons:Dict[str,str], logger:logging.Logger) -> None:
        super().__init__() 
        self.node_id = node_id
        self.motion_definitions = motion_definitons
        self.logger = logger

    def close(self) -> None:
        super().close()

    def handle_event(self, ev:ZoneSequence|TrackDeleted) -> None:
        if isinstance(ev, ZoneSequence):
            seq = ''.join([zone.zone_id for zone in ev])
            seq = ''.join(i for i, _ in itertools.groupby(seq))
            motion_id = self.motion_definitions.get(seq)
            if motion_id:
                frame_index_range = utils.seq_to_range((ev.first_frame_index, ev.frame_index))
                ts_range = utils.seq_to_range((ev.first_ts, ev.ts))
                motion = Motion(node_id=self.node_id, track_id=ev.track_id, motion=motion_id)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'detect motion: track={ev.track_id}, seq={seq}, motion={motion.motion}, '
                                      f'frame={ev.frame_index}')
                self._publish_event(motion)
            else:
                if self.logger.isEnabledFor(logging.WARN):
                    self.logger.warn(f'unknown motion: track={ev.track_id}, seq={seq}, frame={ev.frame_index}')
        else:
            self._publish_event(ev)
    