from __future__ import annotations
from typing import List, Union, Optional, Any
import dataclasses
import threading

import logging
import time
from datetime import timedelta
from omegaconf import OmegaConf

from dna import Frame
from dna.tracker import TrackProcessor, ObjectTrack, TrackState
from dna.tracker.dna_tracker import DNATracker
from .types import TimeElapsed, TrackEvent
from .event_processor import EventQueue, EventListener, EventProcessor
from .event_processors import DropEventByType
from .refine_track_event import RefineTrackEvent

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
            

class PipelineEventPublisher(EventProcessor):
    def handle_event(self, ev:Any) -> None:
        self._publish_event(ev)
        
        
class LogTrackEventPipeline(EventQueue):
    __slots__ = ('plugins')

    def __init__(self, ) -> None:
        super().__init__()

        self.plugins = dict()
        self.sorted_event_queue = self

    def close(self) -> None:
        for plugin in self.plugins.values():
            if hasattr(plugin, 'close') and callable(plugin.close):
                plugin.close()
        super().close()

    def add_plugin(self, id:str, plugin:EventListener, queue:Optional[EventQueue]=None) -> None:
        queue = queue if queue else self
        queue.add_listener(plugin)
        self.plugins[id] = plugin



class TrackEventPipeline(EventQueue):
    __slots__ = ('_input_queue', '_last_queue', 'plugins', 'tick_gen', 'sorted_event_queue')

    def __init__(self, node_id:str, publishing_conf: OmegaConf, load_plugins:bool=True) -> None:
        super().__init__()

        self.node_id = node_id
        self._input_queue = EventQueue()
        self._last_queue = self._input_queue
        self.plugins = dict()
        self.tick_gen = None
        self.sorted_event_queue = self._last_queue
        
        self._output_queue = PipelineEventPublisher()
        self._last_queue.add_listener(self._output_queue)
        
        # drop unnecessary tracks (eg. trailing 'TemporarilyLost' tracks)
        refind_track_conf = OmegaConf.select(publishing_conf, 'refine_tracks', default=None)
        if refind_track_conf:
            buffer_size = OmegaConf.select(refind_track_conf, 'buffer_size', default=_DEFAULT_BUFFER_SIZE)
            buffer_timeout = OmegaConf.select(refind_track_conf, 'buffer_timeout', default=_DEFAULT_BUFFER_TIMEOUT)
            self._append_processor(RefineTrackEvent(buffer_size=buffer_size, buffer_timeout=buffer_timeout))

        # drop too-short tracks of an object
        self.min_path_length = OmegaConf.select(publishing_conf, 'min_path_length', default=_DEFAULT_MIN_PATH_LENGTH)
        if self.min_path_length > 0:
            from .drop_short_trail import DropShortTrail
            self._append_processor(DropShortTrail(self.min_path_length))

        # attach world-coordinates to each track
        if OmegaConf.select(publishing_conf, 'attach_world_coordinates', default=None):
            from .world_coord_attach import WorldCoordinateAttacher
            self._append_processor(WorldCoordinateAttacher(publishing_conf.attach_world_coordinates))

        if OmegaConf.select(publishing_conf, 'stabilization', default=None):
            from .stabilizer import Stabilizer
            self._append_processor(Stabilizer(publishing_conf.stabilization))
        self._append_processor(_DROP_TIME_ELAPSED)
            
        conf = OmegaConf.select(publishing_conf, "order_by_frame_index", default=None)
        if conf:
            from .event_processors import order_by_frame_index
            max_pending_frames = OmegaConf.select(conf, 'max_pending_frames', default=30)
            timeout = OmegaConf.select(refind_track_conf, 'timeout', default=5.0)
            self.sorted_event_queue = order_by_frame_index(self._last_queue, max_pending_frames, timeout)
        else:
            self.sorted_event_queue = self._last_queue

        tick_interval = OmegaConf.select(publishing_conf, 'tick_interval', default=-1)
        if tick_interval > 0:
            self.tick_gen = TimeElapsedGenerator(timedelta(seconds=tick_interval), self._input_queue)
            self.tick_gen.start()

    def close(self) -> None:
        if self.tick_gen:
            self.tick_gen.stop()
        self._input_queue.close()
        for plugin in self.plugins.values():
            if hasattr(plugin, 'close') and callable(plugin.close):
                plugin.close()
        super().close()

    def add_listener(self, listener:EventListener) -> None:
        self._output_queue.add_listener(listener)

    def remove_listener(self, listener:EventListener) -> bool:
        return self._output_queue.remove_listener(listener)

    def _publish_event(self, ev:object) -> None:
        self._output_queue._publish_event(ev)
            
    def handle_event(self, track:TrackEvent) -> None:
        """TrackEvent pipeline으로 입력으로 주어진 track event를 입력시킨다.

        Args:
            track (TrackEvent): 입력 TrackEvent
        """
        self._input_queue._publish_event(track)

    def add_plugin(self, id:str, plugin:EventListener, queue:Optional[EventQueue]=None) -> None:
        queue = queue if queue else self
        queue.add_listener(plugin)
        self.plugins[id] = plugin
        
    def track_started(self, tracker) -> None: pass
    def track_stopped(self, tracker) -> None:
        self.close()
        
    def process_tracks(self, tracker:DNATracker, frame:Frame, tracks:List[ObjectTrack]) -> None:
        for ev in tracker.last_event_tracks:
            ev = dataclasses.replace(ev, node_id=self.node_id)
            self._publish_event(ev)

    def _append_processor(self, proc:EventProcessor) -> None:
        listeners = self._last_queue.listeners.copy()
        for listener in listeners:
            self._last_queue.remove_listener(listener)
        self._last_queue.add_listener(proc)
        
        self._last_queue = proc
        for listener in listeners:
            self._last_queue.add_listener(listener)
            
    def _load_plugins(self, plugins_conf:OmegaConf) -> None:
        cvs_file = OmegaConf.select(plugins_conf, 'output_csv')
        if cvs_file is not None:
            from .utils import CsvTrackEventWriter
            writer = CsvTrackEventWriter(cvs_file)
            self.add_plugin('output_csv', writer, self.sorted_event_queue)
            
        json_file = OmegaConf.select(plugins_conf, 'output')
        if json_file is not None:
            from .utils import JsonTrackEventWriter
            writer = JsonTrackEventWriter(json_file)
            self.add_plugin('output', writer, self.sorted_event_queue)
            
        zone_pipeline_conf = OmegaConf.select(plugins_conf, 'zone_pipeline')
        if zone_pipeline_conf is not None:
            from .zone.zone_pipeline import ZonePipeline
            self.add_plugin('zone_pipeline', ZonePipeline(zone_pipeline_conf))

        local_path_conf = OmegaConf.select(plugins_conf, 'local_path')
        if local_path_conf is not None:
            from .local_path_generator import LocalPathGenerator
            self.add_plugin('local_path', LocalPathGenerator(local_path_conf))
            
        kafka_conf = OmegaConf.select(plugins_conf, 'kafka')
        if kafka_conf is not None:
            from .kafka_event_publisher import KafkaEventPublisher
            self.add_plugin('kafka', KafkaEventPublisher(kafka_conf))
        else:
            import logging
            logger = logging.getLogger('dna.node.kafka')
            logger.warning(f'Kafka publishing is not specified')


def load_plugins(pipeline:TrackEventPipeline, plugins_conf:OmegaConf) -> None:
    cvs_file = OmegaConf.select(plugins_conf, 'output_csv')
    if cvs_file is not None:
        from .utils import CsvTrackEventWriter
        writer = CsvTrackEventWriter(cvs_file)
        pipeline.add_plugin('output_csv', writer, pipeline.sorted_event_queue)
        
    json_file = OmegaConf.select(plugins_conf, 'output')
    if json_file is not None:
        from .utils import JsonTrackEventWriter
        writer = JsonTrackEventWriter(json_file)
        pipeline.add_plugin('output', writer, pipeline.sorted_event_queue)
        
    zone_pipeline_conf = OmegaConf.select(plugins_conf, 'zone_pipeline')
    if zone_pipeline_conf is not None:
        from .zone.zone_pipeline import ZonePipeline
        pipeline.add_plugin('zone_pipeline', ZonePipeline(zone_pipeline_conf))

    local_path_conf = OmegaConf.select(plugins_conf, 'local_path')
    if local_path_conf is not None:
        from .local_path_generator import LocalPathGenerator
        pipeline.add_plugin('local_path', LocalPathGenerator(local_path_conf))
        
    kafka_conf = OmegaConf.select(plugins_conf, 'kafka')
    if kafka_conf is not None:
        from .kafka_event_publisher import KafkaEventPublisher
        pipeline.add_plugin('kafka', KafkaEventPublisher(kafka_conf))
    else:
        import logging
        logger = logging.getLogger('dna.node.kafka')
        logger.warning(f'Kafka publishing is not specified')