from __future__ import annotations
from typing import Tuple, List, Dict, Set, Optional, Any, Union

import threading
from dataclasses import replace

import time
from datetime import timedelta

from omegaconf import OmegaConf

from dna.tracker import ObjectTracker, TrackState, TrackProcessor
from ..types import TimeElapsed
from .types import TrackDeleted
from ..event_processor import EventQueue, EventListener, EventProcessor

import logging


class ZonePipeline(EventListener):
    __slots__ = ( 'event_source', 'event_queues', 'services' )
    
    LOGGER = logging.getLogger('dna.node.zone')
    
    def __init__(self, conf:OmegaConf) -> None:
        super().__init__()
        
        self.event_source = EventQueue()
        self.event_queues:Dict[str,EventQueue] = dict()
        self.services:Dict[str,Any] = dict()
        
        from .to_line_transform import ToLineTransform
        to_line = ToLineTransform(ZonePipeline.LOGGER.getChild('line'))
        self.event_source.add_listener(to_line)
        self.line_event_queue = to_line
        
        named_zones = OmegaConf.select(conf, "zones", default=[])
        from .zone_event_generator import ZoneEventGenerator
        zone_detector = ZoneEventGenerator(named_zones, ZonePipeline.LOGGER.getChild('zone'))
        self.line_event_queue.add_listener(zone_detector)
        self._raw_zone_event_queue = zone_detector
        
        from .zone_event_refiner import ZoneEventRefiner
        event_refiner = ZoneEventRefiner(ZonePipeline.LOGGER.getChild('zone'))
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
        
        motions = OmegaConf.select(conf, "motions")
        if motions:
            from .motion_detector import MotionDetector
            motion = MotionDetector(dict(motions), logger=ZonePipeline.LOGGER.getChild('motion'))
            last_zone_seq.add_listener(motion)
            self.services['motions'] = motion
            self.event_queues['motions'] = motion

    def close(self) -> None:
        self.event_source.close()
    
    def handle_event(self, ev: object) -> None:
        self.event_source.publish_event(ev)


from dna.node import TrackEvent
from dna.node.zone import ZoneEvent
class ToTrackEvent(EventProcessor):
    def handle_event(self, ev:ZoneEvent) -> None:
        if ev.source:
            self.publish_event(ev.to_track_event())