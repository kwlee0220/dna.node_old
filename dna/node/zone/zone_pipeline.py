from __future__ import annotations
from typing import Tuple, List, Dict, Set, Optional, Any
import threading

import time
from datetime import timedelta

from omegaconf import OmegaConf

from dna.tracker import ObjectTracker, TrackState, TrackProcessor
from ..event_processor import EventQueue, EventListener, EventProcessor
from .types import TimeElapsed

import logging
LOGGER = logging.getLogger('dna.node.zone')

class TimeElapsedGenerator(threading.Thread):
    def __init__(self, interval:timedelta, publishing_queue:EventQueue):
        threading.Thread.__init__(self)
        self.daemon = False
        self.stopped = threading.Event()
        self.interval = interval
        self.publishing_queue = publishing_queue
        
    def stop(self):
        self.stopped.set()
        self.join()
        
    def run(self):
        while not self.stopped.wait(self.interval.total_seconds()):
            self.publishing_queue.publish_event(TimeElapsed(frame_index=-1, ts=time.time()))


class ZonePipeline(EventListener):
    __slots__ = ( 'event_queues', 'resident_change_detector' )

    def __init__(self, conf:OmegaConf) -> None:
        super().__init__()
        
        self.event_source = EventQueue()
        self.event_queues:Dict[str,EventQueue] = dict()
        self.services:Dict[str,Any] = dict()
        
        from .to_line_transform import ToLineTransform
        to_line = ToLineTransform(LOGGER.getChild('line'))
        self.event_source.add_listener(to_line)
        self.line_event_queue = to_line
        
        named_zones = OmegaConf.select(conf, "zones", default=[])
        from .zone_event_generator import ZoneEventGenerator
        zone_detector = ZoneEventGenerator(named_zones, LOGGER.getChild('zone'))
        self.line_event_queue.add_listener(zone_detector)
        self._raw_zone_event_queue = zone_detector
        
        from .zone_event_refiner import ZoneEventRefiner
        event_refiner = ZoneEventRefiner(LOGGER.getChild('zone'))
        zone_detector.add_listener(event_refiner)
        self.event_queues['zone_events'] = event_refiner.zone_event_queue
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
        filter = FinalZoneSequenceFilter()
        collector.add_listener(filter)
        
        motions = OmegaConf.select(conf, "motions")
        if motions:
            from .motion_detector import MotionDetector
            motion = MotionDetector(dict(motions), logger=LOGGER.getChild('motion'))
            filter.add_listener(motion)
            self.services['motions'] = motion
            self.event_queues['motions'] = motion
        
            # from ..event_processor import PrintEvent
            # print = PrintEvent()
            # motion.add_listener(print)

    def close(self) -> None:
        self.event_source.close()
    
    def handle_event(self, ev: object) -> None:
        self.event_source.publish_event(ev)