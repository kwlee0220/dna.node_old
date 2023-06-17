from __future__ import annotations

from typing import Union, Optional
import dataclasses
import threading

import logging
import numpy as np
from datetime import timedelta
from omegaconf import OmegaConf

from dna import Frame, Size2d, config, Box, Point
from dna.camera import ImageProcessor
from dna.event import TrackEvent, TimeElapsed, TrackDeleted, EventQueue, EventListener, EventProcessor
from dna.event.event_processors import DropEventByType, GroupByFrameIndex, EventRelay, TimeElapsedGenerator
from dna.track import TrackState
from dna.track.types import TrackProcessor, ObjectTrack
from dna.track.dna_tracker import DNATracker
from .zone.zone_pipeline import ZonePipeline

_DEFAULT_BUFFER_SIZE = 30
_DEFAULT_BUFFER_TIMEOUT = 5.0
# _DEFAULT_MIN_PATH_LENGTH=10


# class TimeElapsedGenerator(threading.Thread):
#     def __init__(self, interval:timedelta, publishing_queue:EventQueue):
#         threading.Thread.__init__(self)
#         self.daemon = False
#         self.stopped = threading.Event()
#         self.interval = interval
#         self.publishing_queue = publishing_queue
        
#     def stop(self):
#         self.stopped.set()
#         self.join()
        
#     def run(self):
#         while not self.stopped.wait(self.interval.total_seconds()):
#             self.publishing_queue._publish_event(TimeElapsed())
        
        
class LogTrackEventPipeline(EventQueue):
    __slots__ = ('plugins')

    def __init__(self, track_file) -> None:
        super().__init__()

        self.track_file = track_file
        self.plugins = dict()
        self.sorted_event_queue = self

    def close(self) -> None:
        for plugin in self.plugins.values():
            if hasattr(plugin, 'close') and callable(plugin.close):
                plugin.close()
        super().close()
    
    @property
    def group_event_queue(self) -> EventQueue:
        if not self._group_event_queue:
            from ..event.event_processors import GroupByFrameIndex
            self._group_event_queue = GroupByFrameIndex(max_pending_frames=1, timeout=0.5)
            self.add_listener(self._group_event_queue)
        return self._group_event_queue

    def add_plugin(self, id:str, plugin:EventListener, queue:Optional[EventQueue]=None) -> None:
        queue = queue if queue else self
        queue.add_listener(plugin)
        self.plugins[id] = plugin
        
    def run(self) -> None:
        from dna.event.utils import read_tracks_json
        for track in read_tracks_json(self.track_file):
            self._publish_event(track)
        self.close()


class MinFrameIndexComposer:
    def __init__(self) -> None:
        self.processors:list[EventProcessor] = []
        self.min_indexes:list[int] = []
        self.min_holder = -1
        
    def append(self, proc:EventProcessor) -> None:
        self.processors.append(proc)
        self.min_indexes.append(None)
        
    def min_frame_index(self) -> int:
        import sys
        
        if self.min_holder >= 0:
            min = self.processors[self.min_holder].min_frame_index()
            if min == self.min_indexes[self.min_holder]:
                return min
        for idx, proc in enumerate(self.processors):
            min = proc.min_frame_index()
            self.min_indexes[idx] = min if min else sys.maxsize
        
        self.min_holder = np.argmin(self.min_indexes)
        min = self.min_indexes[self.min_holder]
        if min != sys.maxsize:
            return min
        else:
            self.min_holder = -1
            return None


class TrackEventPipeline(EventQueue,TrackProcessor):
    __slots__ = ('node_id', 'plugins', '_tick_gen', '_input_queue', '_output_queue',
                 '_current_queue', '_group_event_queue', 'min_frame_indexers')

    def __init__(self, node_id:str, publishing_conf:OmegaConf,
                 image_processor:Optional[ImageProcessor]=None) -> None:
        EventQueue.__init__(self)
        TrackProcessor.__init__(self)

        self.node_id = node_id
        self.plugins = dict()
        self._tick_gen = None
        
        self._input_queue = EventQueue()
        self._event_publisher = EventRelay(self)
        self._current_queue = self._input_queue
        self._current_queue.add_listener(self._event_publisher)
        self._group_event_queue:GroupByFrameIndex = None
        
        self.logger = logging.getLogger("dna.node.event")
        
        self.min_frame_indexers:MinFrameIndexComposer = MinFrameIndexComposer()
        
        # drop unnecessary tracks (eg. trailing 'TemporarilyLost' tracks)
        refind_track_conf = config.get(publishing_conf, 'refine_tracks')
        if refind_track_conf:
            from .refine_track_event import RefineTrackEvent
            buffer_size = config.get(refind_track_conf, 'buffer_size', default=_DEFAULT_BUFFER_SIZE)
            buffer_timeout = config.get(refind_track_conf, 'buffer_timeout', default=_DEFAULT_BUFFER_TIMEOUT)
            refine_tracks = RefineTrackEvent(buffer_size=buffer_size, buffer_timeout=buffer_timeout,
                                                   logger=self.logger)
            self._append_processor(refine_tracks)
            self.min_frame_indexers.append(refine_tracks)

        # drop too-short tracks of an object
        min_path_length = config.get(publishing_conf, 'min_path_length', default=-1)
        if min_path_length > 0:
            from .drop_short_trail import DropShortTrail
            drop_short_trail = DropShortTrail(min_path_length, logger=self.logger)
            self._append_processor(drop_short_trail)
            self.min_frame_indexers.append(drop_short_trail)

        # attach world-coordinates to each track
        if config.exists(publishing_conf, 'attach_world_coordinates'):
            from .world_coord_attach import WorldCoordinateAttacher
            self._append_processor(WorldCoordinateAttacher(publishing_conf.attach_world_coordinates))

        if config.exists(publishing_conf, 'stabilization'):
            from .stabilizer import Stabilizer
            stabilizer = Stabilizer(publishing_conf.stabilization)
            self._append_processor(stabilizer)
            self.min_frame_indexers.append(stabilizer)
        self._append_processor(DropEventByType([TimeElapsed]))
        
        # generate zone-based events
        zone_pipeline_conf = config.get(publishing_conf, 'zone_pipeline')
        if zone_pipeline_conf:
            zone_pipeline = ZonePipeline(self.node_id, zone_pipeline_conf)
            self._current_queue.add_listener(zone_pipeline)
            self.plugins['zone_pipeline'] = zone_pipeline
            
            transform = ZoneToTrackEventTransform()
            self._current_queue.remove_listener(self._event_publisher)
            transform.add_listener(self._event_publisher)
            self._current_queue = transform
            zone_pipeline.event_queues['zone_events'].add_listener(transform)
    
        # 알려진 TrackEventPipeline의 plugin 을 생성하여 등록시킨다.
        plugins_conf = config.get(publishing_conf, "plugins")
        if plugins_conf:
            load_plugins(plugins_conf, self, image_processor, logger=self.logger)

        tick_interval = config.get(publishing_conf, 'tick_interval', default=-1)
        if tick_interval > 0:
            self._tick_gen = TimeElapsedGenerator(timedelta(seconds=tick_interval))
            self._tick_gen.add_listener(self)
            self._tick_gen.start()

    def close(self) -> None:
        if self._tick_gen:
            self._tick_gen.stop()
        self._input_queue.close()
        
        super().close()
        
        for plugin in reversed(self.plugins.values()):
            if hasattr(plugin, 'close') and callable(plugin.close):
                plugin.close()
            
    def handle_event(self, track:TrackEvent) -> None:
        """TrackEvent pipeline에 주어진 track event를 입력시킨다.

        Args:
            track (TrackEvent): 입력 TrackEvent
        """
        self._input_queue._publish_event(track)
    
    @property
    def group_event_queue(self) -> EventQueue:
        if not self._group_event_queue:
            from ..event.event_processors import GroupByFrameIndex
            self._group_event_queue = GroupByFrameIndex(self.min_frame_indexers.min_frame_index)
            self.add_listener(self._group_event_queue)
        return self._group_event_queue
        
    def track_started(self, tracker) -> None: pass
    def track_stopped(self, tracker) -> None:
        self.close()
        
    def process_tracks(self, tracker:DNATracker, frame:Frame, tracks:list[ObjectTrack]) -> None:
        for ev in tracker.last_event_tracks:
            ev = dataclasses.replace(ev, node_id=self.node_id)
            self.handle_event(ev)

    def _append_processor(self, proc:EventProcessor) -> None:
        self._current_queue.remove_listener(self._event_publisher)
        proc.add_listener(self._event_publisher)
        self._current_queue.add_listener(proc)
        self._current_queue = proc
        
        
from dataclasses import replace
from .zone import ZoneEvent
class ZoneToTrackEventTransform(EventProcessor):
    def handle_event(self, ev:Union[ZoneEvent,TrackDeleted]) -> None:
        if isinstance(ev, ZoneEvent):
            if ev.source:
                track_ev = replace(ev.source, zone_relation=ev.relation_str())
                self._publish_event(track_ev)
        elif isinstance(ev, TrackDeleted):
            if ev.source:
                track_ev = replace(ev.source, zone_relation='D')
                self._publish_event(track_ev)


def load_plugins(plugins_conf:OmegaConf, pipeline:TrackEventPipeline,
                 image_processor:Optional[ImageProcessor]=None,
                 *, logger:Optional[logging.Logger]=None) -> None:
    local_path_conf = config.get(plugins_conf, 'local_path')
    if local_path_conf:
        from .local_path_generator import LocalPathGenerator
        plugin = LocalPathGenerator(local_path_conf)
        pipeline.add_listener(plugin)
        pipeline.plugins['local_path'] = plugin
        
    default_kafka_brokers = config.get(plugins_conf, 'kafka_brokers')
        
    publish_tracks_conf = config.get(plugins_conf, 'publish_tracks')
    if publish_tracks_conf:
        from dna.event import KafkaEventPublisher
        
        kafka_brokers = config.get(publish_tracks_conf, 'kafka_brokers', default=default_kafka_brokers)
        topic = config.get(publish_tracks_conf, 'topic', default='track-events')
        plugin = KafkaEventPublisher(kafka_brokers=kafka_brokers, topic=topic, logger=logger.getChild('kafka.tracks'))
        pipeline.add_listener(plugin)
        pipeline.plugins['publish_tracks'] = plugin
            
    # 'PublishReIDFeatures' plugin은 ImageProcessor가 지정된 경우에만 등록시킴
    publish_features_conf = config.get(plugins_conf, "publish_features")
    if publish_features_conf and image_processor:
        from dna.track.dna_tracker import load_feature_extractor
        from .reid_features import PublishReIDFeatures
        
        distinct_distance = publish_features_conf.get('distinct_distance', 0.0)
        min_crop_size = Size2d.from_expr(publish_features_conf.get('min_crop_size', '80x80'))
        publish = PublishReIDFeatures(extractor=load_feature_extractor(normalize=True),
                                      distinct_distance=distinct_distance,
                                      min_crop_size=min_crop_size)
        pipeline.group_event_queue.add_listener(publish)
        image_processor.add_frame_processor(publish)
        
        kafka_brokers = config.get(publish_features_conf, 'kafka_brokers', default=default_kafka_brokers)
        topic = config.get(publish_features_conf, 'topic', default='track-features')
        plugin = KafkaEventPublisher(kafka_brokers=kafka_brokers, topic=topic, logger=logger.getChild('kafka.features'))
        publish.add_listener(plugin)
        pipeline.plugins['publish_features'] = plugin
        
    zone_pipeline:ZonePipeline = pipeline.plugins.get('zone_pipeline')
    if zone_pipeline:
        publish_motions_conf = config.get(plugins_conf, 'publish_motions')
        if publish_motions_conf:
            from ..event.kafka_event_publisher import KafkaEventPublisher
            motions = zone_pipeline.event_queues.get('motions')
            if motions:
                kafka_brokers = config.get(publish_motions_conf, 'kafka_brokers', default=default_kafka_brokers)
                topic = config.get(publish_motions_conf, 'topic', default='track-motions')
                plugin = KafkaEventPublisher(kafka_brokers=kafka_brokers, topic=topic, logger=logger.getChild('kafka.motions'))
                motions.add_listener(plugin)
                pipeline.plugins['publish_motions'] = plugin
    
    output_file = config.get(plugins_conf, 'output')
    if output_file is not None:
        from .utils import JsonTrackEventGroupWriter
        writer = JsonTrackEventGroupWriter(output_file)
        pipeline.group_event_queue.add_listener(writer)
        pipeline.plugins['output'] = writer