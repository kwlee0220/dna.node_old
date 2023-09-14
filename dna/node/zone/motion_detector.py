from __future__ import annotations

from typing import Optional
import itertools
import logging

from dna.event import EventProcessor, TrackletMotion
from .zone_sequence_collector import ZoneSequence
    

class MotionDetector(EventProcessor):
    def __init__(self, motion_definitons:dict[str,str], *,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__() 
        self.motion_definitions = motion_definitons
        self.logger = logger

    def close(self) -> None:
        super().close()

    def handle_event(self, ev:ZoneSequence) -> None:
        if isinstance(ev, ZoneSequence):
            seq = ''.join([visit.zone_id for visit in ev])
            seq = ''.join(i for i, _ in itertools.groupby(seq))
            seq_str = ev.sequence_str()
            enter_zone = seq[0] if seq else None
            exit_zone = seq[-1] if seq else None
            motion_id = self.motion_definitions.get(seq) if self.motion_definitions else None
            
            if motion_id:
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'detect motion: track={ev.track_id}, seq={seq_str}, motion={motion.motion}, '
                                        f'frame={ev.frame_index}')
            elif self.motion_definitions:
                # motion 정보가 등록되어 있음에도 motion 검출에 실패한 경우.
                if self.logger:
                    self.logger.warn(f'unknown motion: track={ev.track_id}, seq={seq_str}, frame={ev.frame_index}')
                    
            motion = TrackletMotion(node_id=ev.node_id,
                                    track_id=ev.track_id,
                                    zone_sequence=seq_str,
                                    enter_zone=enter_zone,
                                    exit_zone=exit_zone,
                                    motion=motion_id,
                                    frame_index=ev.frame_index,
                                    ts=ev.ts)
            self._publish_event(motion)
        else:
            self._publish_event(ev)
    