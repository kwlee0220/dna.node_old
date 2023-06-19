from __future__ import annotations

from typing import Optional
import logging

from omegaconf import OmegaConf

from dna import config, sub_logger
from dna.event import EventQueue, EventListener
from dna.event.event_processors import EventRelay
from .types import ZoneEvent


class ZonePipeline(EventListener,EventQueue):
    def __init__(self, node_id:str, conf:OmegaConf,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        
        self.node_id = node_id
        self.event_source = EventQueue()
        self.event_queues:dict[str,EventQueue] = dict()
        
        from .to_line_transform import ToLineTransform
        to_line = ToLineTransform(logger=sub_logger(logger, 'line'))
        self.event_source.add_listener(to_line)
        self.line_event_queue = to_line
        
        from .zone_event_generator import ZoneEventGenerator
        named_zones = config.get(conf, "zones", default=[])
        zone_detector = ZoneEventGenerator(named_zones, logger=sub_logger(logger, 'zone_gen'))
        self.line_event_queue.add_listener(zone_detector)
        self._raw_zone_event_queue = zone_detector
        
        from .zone_event_refiner import ZoneEventRefiner
        event_refiner = ZoneEventRefiner(logger=sub_logger(logger, 'zone_refine'))
        event_refiner.add_listener(EventRelay(self))
        zone_detector.add_listener(event_refiner)
        self.event_queues['zone_events'] = event_refiner
        self.event_queues['location_changes'] = event_refiner.location_event_queue
        
        from .resident_changes import ResidentChanges
        self.resident_changes = ResidentChanges()
        self.event_queues['zone_events'].add_listener(self.resident_changes)
        self.event_queues['resident_changes'] = self.resident_changes
        
        from .zone_sequence_collector import ZoneSequenceCollector
        collector = ZoneSequenceCollector()
        self.event_queues['zone_events'].add_listener(collector)
        self.event_queues['zone_sequences'] = collector
        
        from .zone_sequence_collector import FinalZoneSequenceFilter
        last_zone_seq = FinalZoneSequenceFilter()
        self.event_queues['zone_sequences'].add_listener(last_zone_seq)
        self.event_queues['last_zone_sequences'] = last_zone_seq
        
        if motions := config.get(conf, "motions"):
            from .motion_detector import MotionDetector
            motion = MotionDetector(self.node_id, dict(motions), logger=sub_logger(logger, 'motion'))
            last_zone_seq.add_listener(motion)
            self.event_queues['motions'] = motion

    def close(self) -> None:
        self.event_source.close()
    
    def handle_event(self, ev: object) -> None:
        self.event_source._publish_event(ev)