from __future__ import annotations
from typing import List, Tuple, Generator
import dataclasses

from omegaconf import OmegaConf

from dna import Frame
from dna.tracker import TrackProcessor, ObjectTrack, TrackState
from dna.tracker.dna_tracker import DNATracker
from .event_processor import EventQueue, EventListener
from .refine_track_event import RefineTrackEvent


_DEFAULT_BUFFER_SIZE = 30
_DEFAULT_MIN_PATH_LENGTH=10

class TrackEventPipeline(TrackProcessor,EventQueue):
    __slots__ = ('node_id', 'track_event_queue', 'refine_buffer_size', 'min_path_length', 'plugins')

    def __init__(self, node_id: str, publishing_conf: OmegaConf) -> None:
        super().__init__()

        self.node_id = node_id
        self._event_source = EventQueue()
        self.track_event_queue = self._event_source
        
        # drop unnecessary tracks (eg. trailing 'TemporarilyLost' tracks)
        self.refine_buffer_size = publishing_conf.get('refine_buffer_size', _DEFAULT_BUFFER_SIZE)
        refine = RefineTrackEvent(self.refine_buffer_size)
        self.track_event_queue.add_listener(refine)
        self.track_event_queue = refine

        # drop too-short tracks of an object
        self.min_path_length = publishing_conf.get('min_path_length', _DEFAULT_MIN_PATH_LENGTH)
        if self.min_path_length > 0:
            from .drop_short_trail import DropShortTrail
            drop_short_path = DropShortTrail(self.min_path_length)
            self.track_event_queue.add_listener(drop_short_path)
            self.track_event_queue = drop_short_path

        # attach world-coordinates to each track
        if publishing_conf.get('attach_world_coordinates') is not None:
            from .world_coord_attach import WorldCoordinateAttacher
            attacher = WorldCoordinateAttacher(publishing_conf.attach_world_coordinates)
            self.track_event_queue.add_listener(attacher)
            self.track_event_queue = attacher

        if publishing_conf.get('stabilization') is not None:
            from .stabilizer import Stabilizer
            stabilizer = Stabilizer(publishing_conf.stabilization)
            self.track_event_queue.add_listener(stabilizer)
            self.track_event_queue = stabilizer
            
        if OmegaConf.select(publishing_conf, "sort_by_frame_index", default=False):
            from .event_processors import GroupByFrameIndex, UngroupTrackEvents
            groupby_frame = GroupByFrameIndex(max_delay_frames=self.refine_buffer_size)
            ungroup = UngroupTrackEvents()
            self.track_event_queue.add_listener(groupby_frame)
            groupby_frame.add_listener(ungroup)
            self.track_event_queue = ungroup

        plugins_conf = OmegaConf.select(publishing_conf, 'plugins')
        if plugins_conf is not None:
            self.plugins = {id:plugin for id, plugin in self.load_plugins(plugins_conf)}
            for plugin in self.plugins.values():
                self.track_event_queue.add_listener(plugin)
        else:
            self.plugins = dict()
            
    def load_plugins(self, plugins_conf:OmegaConf) -> Generator[Tuple[str,EventListener], None, None]:
        cvs_file = OmegaConf.select(plugins_conf, 'output_csv')
        if cvs_file is not None:
            from .track_event_writer import CsvTrackEventWriter
            yield 'output_csv', CsvTrackEventWriter(cvs_file)
            
        json_file = OmegaConf.select(plugins_conf, 'output')
        if json_file is not None:
            from .track_event_writer import JsonTrackEventWriter
            yield 'output_csv', JsonTrackEventWriter(json_file)
            
        zone_pipeline_conf = OmegaConf.select(plugins_conf, 'zone_pipeline')
        if zone_pipeline_conf is not None:
            from .zone.zone_pipeline import ZonePipeline
            yield 'zone_pipeline', ZonePipeline(zone_pipeline_conf)

        local_path_conf = OmegaConf.select(plugins_conf, 'local_path')
        if local_path_conf is not None:
            from .local_path_generator import LocalPathGenerator
            yield 'local_path', LocalPathGenerator(local_path_conf)
            
        kafka_conf = OmegaConf.select(plugins_conf, 'kafka')
        if kafka_conf is not None:
            from .kafka_event_publisher import KafkaEventPublisher
            yield 'kafka', KafkaEventPublisher(kafka_conf)
        else:
            import logging
            logger = logging.getLogger('dna.node.kafka')
            logger.warning(f'Kafka publishing is not specified')

    def add_listener(self, listener:EventListener) -> None:
        self.track_event_queue.listeners.append(listener)

    def publish_event(self, ev:object) -> None:
        self.track_event_queue.publish_event(ev)

    def track_started(self, tracker) -> None: pass
    def track_stopped(self, tracker) -> None:
        self._event_source.close()
        
    def process_tracks(self, tracker:DNATracker, frame:Frame, tracks:List[ObjectTrack]) -> None:
        for ev in tracker.last_event_tracks:
            ev = dataclasses.replace(ev, node_id=self.node_id)
            self._event_source.publish_event(ev)