from __future__ import annotations
from typing import List, Union, Optional, Any
import dataclasses
import threading

import logging
import time
from datetime import timedelta
from omegaconf import OmegaConf

from dna import Frame, Size2d
from dna.camera import ImageProcessor
from dna.tracker import TrackProcessor, ObjectTrack, TrackState
from dna.tracker.dna_tracker import DNATracker
from .types import TimeElapsed, TrackEvent
from .event_processor import EventQueue, EventListener, EventProcessor
from .event_processors import DropEventByType, GroupByFrameIndex, EventRelay
from .zone.zone_pipeline import ZonePipeline

_DEFAULT_BUFFER_SIZE = 30
_DEFAULT_BUFFER_TIMEOUT = 5.0
_DEFAULT_MIN_PATH_LENGTH=10

LOGGER = logging.getLogger('dna.node.event')


_DROP_TIME_ELAPSED = DropEventByType([TimeElapsed])


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
            self.publishing_queue._publish_event(TimeElapsed())
        
        
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
            from .event_processors import GroupByFrameIndex
            self._group_event_queue = GroupByFrameIndex(max_pending_frames=1, timeout=0.5)
            self.add_listener(self._group_event_queue)
        return self._group_event_queue

    def add_plugin(self, id:str, plugin:EventListener, queue:Optional[EventQueue]=None) -> None:
        queue = queue if queue else self
        queue.add_listener(plugin)
        self.plugins[id] = plugin
        
    def run(self) -> None:
        from dna.node.utils import read_tracks_json
        for track in read_tracks_json(self.track_file):
            self._publish_event(track)
        self.close()


class TrackEventPipeline(EventQueue):
    __slots__ = ('node_id', 'plugins', '_tick_gen', '_input_queue', '_output_queue',
                 '_last_queue', '_group_event_queue')

    def __init__(self, node_id:str, publishing_conf: OmegaConf,
                 image_processor:Optional[ImageProcessor]=None) -> None:
        super().__init__()

        self.node_id = node_id
        self.plugins = dict()
        self._tick_gen = None
        
        self._input_queue = EventQueue()
        self._last_queue = self._input_queue
        self._event_publisher = EventRelay(self)
        self._last_queue.add_listener(self._event_publisher)
        self._group_event_queue:GroupByFrameIndex = None
        
        self._refine_tracks = None
        self._drop_short_trail = None
        
        # drop unnecessary tracks (eg. trailing 'TemporarilyLost' tracks)
        refind_track_conf = OmegaConf.select(publishing_conf, 'refine_tracks', default=None)
        if refind_track_conf:
            from .refine_track_event import RefineTrackEvent
            buffer_size = OmegaConf.select(refind_track_conf, 'buffer_size', default=_DEFAULT_BUFFER_SIZE)
            buffer_timeout = OmegaConf.select(refind_track_conf, 'buffer_timeout', default=_DEFAULT_BUFFER_TIMEOUT)
            self._refine_tracks = RefineTrackEvent(buffer_size=buffer_size, buffer_timeout=buffer_timeout)
            self._append_processor(self._refine_tracks)

        # drop too-short tracks of an object
        self.min_path_length = OmegaConf.select(publishing_conf, 'min_path_length', default=_DEFAULT_MIN_PATH_LENGTH)
        if self.min_path_length > 0:
            from .drop_short_trail import DropShortTrail
            self._drop_short_trail = DropShortTrail(self.min_path_length)
            self._append_processor(self._drop_short_trail)

        # attach world-coordinates to each track
        if OmegaConf.select(publishing_conf, 'attach_world_coordinates', default=None):
            from .world_coord_attach import WorldCoordinateAttacher
            self._append_processor(WorldCoordinateAttacher(publishing_conf.attach_world_coordinates))

        if OmegaConf.select(publishing_conf, 'stabilization', default=None):
            from .stabilizer import Stabilizer
            self._append_processor(Stabilizer(publishing_conf.stabilization))
        self._append_processor(_DROP_TIME_ELAPSED)
        
        # generate zone-based events
        zone_pipeline_conf = OmegaConf.select(publishing_conf, 'zone_pipeline')
        if zone_pipeline_conf:
            zone_pipeline = ZonePipeline(self.node_id, zone_pipeline_conf)
            self._last_queue.add_listener(zone_pipeline)
            self.plugins['zone_pipeline'] = zone_pipeline
            
            self._last_queue.remove_listener(self._event_publisher)
            transform = ZoneToTrackEventTransform()
            transform.add_listener(self._event_publisher)
            self._last_queue = transform
            zone_pipeline.event_queues['zone_events'].add_listener(transform)
    
        # 알려진 TrackEventPipeline의 plugin 을 생성하여 등록시킨다.
        plugins_conf = OmegaConf.select(publishing_conf, "plugins", default=None)
        if plugins_conf:
            load_plugins(plugins_conf, self, image_processor)

        tick_interval = OmegaConf.select(publishing_conf, 'tick_interval', default=-1)
        if tick_interval > 0:
            self._tick_gen = TimeElapsedGenerator(timedelta(seconds=tick_interval), self._input_queue)
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
        """TrackEvent pipeline으로 입력으로 주어진 track event를 입력시킨다.

        Args:
            track (TrackEvent): 입력 TrackEvent
        """
        self._input_queue._publish_event(track)
    
    @property
    def group_event_queue(self) -> EventQueue:
        if not self._group_event_queue:
            from .event_processors import GroupByFrameIndex
            len1 = self._refine_tracks.buffer_size if self._refine_tracks else 0
            len2 = self._drop_short_trail.min_trail_length if self._drop_short_trail else 0
            buffer_size = max(len1, len2)
            self._group_event_queue = GroupByFrameIndex(max_pending_frames=buffer_size, timeout=5.0)
            self.add_listener(self._group_event_queue)
        return self._group_event_queue
        
    def track_started(self, tracker) -> None: pass
    def track_stopped(self, tracker) -> None:
        self.close()
        
    def process_tracks(self, tracker:DNATracker, frame:Frame, tracks:List[ObjectTrack]) -> None:
        for ev in tracker.last_event_tracks:
            ev = dataclasses.replace(ev, node_id=self.node_id)
            # self._publish_event(ev)
            self.handle_event(ev)

    def _append_processor(self, proc:EventProcessor) -> None:
        self._last_queue.remove_listener(self._event_publisher)
        proc.add_listener(self._event_publisher)
        self._last_queue.add_listener(proc)
        self._last_queue = proc
        
        
from dataclasses import replace
from .zone import ZoneEvent, TrackDeleted
class ZoneToTrackEventTransform(EventProcessor):
    def handle_event(self, ev:ZoneEvent|TrackDeleted) -> None:
        if isinstance(ev, ZoneEvent):
            if ev.source:
                track_ev = replace(ev.source, zone_relation=ev.relation_str())
                self._publish_event(track_ev)
        elif isinstance(ev, TrackDeleted):
            if ev.source:
                track_ev = replace(ev.source, zone_relation='D')
                self._publish_event(track_ev)


def load_plugins(plugins_conf:OmegaConf, pipeline:TrackEventPipeline,
                 image_processor:Optional[ImageProcessor]=None) -> None:
    zone_pipeline:ZonePipeline = pipeline.plugins['zone_pipeline']
    
    output_file = OmegaConf.select(plugins_conf, 'output')
    if output_file is not None:
        from .utils import JsonTrackEventGroupWriter
        writer = JsonTrackEventGroupWriter(output_file)
        pipeline.group_event_queue.add_listener(writer)
        pipeline.plugins['output'] = writer

    local_path_conf = OmegaConf.select(plugins_conf, 'local_path')
    if local_path_conf:
        from .local_path_generator import LocalPathGenerator
        plugin = LocalPathGenerator(local_path_conf)
        pipeline.plugins['local_path'] = plugin
        pipeline.add_listener(plugin)
        
    publish_tracks_conf = OmegaConf.select(plugins_conf, 'publish_tracks')
    if publish_tracks_conf:
        from .kafka_event_publisher import KafkaEventPublisher
        plugin = KafkaEventPublisher(publish_tracks_conf)
        pipeline.plugins['publish_tracks'] = plugin
        pipeline.add_listener(plugin)
        
    publish_motions_conf = OmegaConf.select(plugins_conf, 'publish_motions')
    if publish_motions_conf:
        from .kafka_event_publisher import KafkaEventPublisher
        motions = zone_pipeline.event_queues.get('motions')
        if motions:
            plugin = KafkaEventPublisher(publish_motions_conf)
            pipeline.plugins['publish_motions'] = plugin
            motions.add_listener(plugin)
            
    # 'PublishReIDFeatures' plugin은 ImageProcessor가 지정된 경우에만 등록시킴
    publish_features_conf = OmegaConf.select(plugins_conf, "publish_features", default=None)
    if publish_features_conf and image_processor:
        from dna.support.sql_utils import SQLConnector
        from dna.node import TrackletStore
        from dna.tracker.dna_tracker import load_feature_extractor
        from .reid_features import PublishReIDFeatures
        frame_buf_size = pipeline.group_event_queue.max_pending_frames + 2
        distinct_distance = publish_features_conf.get('distinct_distance', 0.0)
        min_crop_size = Size2d.from_expr(publish_features_conf.get('min_crop_size', '80x80'))
        publish = PublishReIDFeatures(frame_buffer_size=frame_buf_size,
                                        extractor=load_feature_extractor(normalize=True),
                                        distinct_distance=distinct_distance,
                                        min_crop_size=min_crop_size)
        pipeline.group_event_queue.add_listener(publish)
        image_processor.add_frame_processor(publish)
        
        # tracklet_store = TrackletStore(SQLConnector.from_conf(publish_features_conf))
        from .kafka_event_publisher import KafkaEventPublisher
        plugin = KafkaEventPublisher(publish_features_conf)
        pipeline.plugins['publish_features'] = plugin
        publish.add_listener(plugin)