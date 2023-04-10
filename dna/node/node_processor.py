from __future__ import annotations
from typing import Optional, List, Tuple

from pathlib import Path
from omegaconf import OmegaConf

from dna.conf import exists_config, load_config
from dna.camera import ImageProcessor, ImageCapture, FrameProcessor
from dna.execution import Execution, ExecutionContext, NoOpExecutionContext
from dna.pika_execution import PikaExecutionContext, PikaExecutionFactory
from dna.tracker.track_pipeline import TrackingPipeline, TrackProcessor
from .event_processor import EventListener
from .track_event_pipeline import TrackEventPipeline, load_plugins
from .zone.zone_pipeline import ZonePipeline
from .zone.zone_sequences_display import ZoneSequenceDisplay


_DEFAULT_EXEC_CONTEXT = NoOpExecutionContext()
 

def build_node_processor(conf: OmegaConf, /,
                         image_processor:Optional[ImageProcessor]=None,
                         tracker_pipeline:Optional[TrackingPipeline]=None,
                         context: ExecutionContext=_DEFAULT_EXEC_CONTEXT) \
    -> Tuple[ImageProcessor, TrackingPipeline, TrackEventPipeline]:
    # TrackingPipeline 생성하고 ImageProcessor에 등록함
    if not tracker_pipeline:
        tracker_conf = conf.get('tracker', OmegaConf.create())
        tracker_pipeline = TrackingPipeline.load(tracker_conf)
    if image_processor:
        image_processor.add_frame_processor(tracker_pipeline)

    # TrackEventPipeline 생성하고 TrackingPipeline에 등록함
    publishing_conf = conf.get('publishing', OmegaConf.create())
    track_event_pipeline = TrackEventPipeline(conf.id, publishing_conf=publishing_conf,
                                              image_processor=image_processor)
    tracker_pipeline.add_track_processor(track_event_pipeline)

    # ZonePipeline이 TrackEventPipeline에 등록되고, motion detection이 정의된 경우
    # 이를 ZonePipeline에 등록시킨다
    zone_pipeline:ZonePipeline = track_event_pipeline.plugins.get('zone_pipeline')
    if zone_pipeline and image_processor and image_processor.is_drawing():
        motion_detector = zone_pipeline.services.get('motions')
        motion_queue = zone_pipeline.event_queues.get('motions')
        if motion_detector and motion_queue:
            display = ZoneSequenceDisplay(motion_definitions=motion_detector.motion_definitions,
                                          track_queue=track_event_pipeline,
                                          motion_queue=motion_queue)
            image_processor.add_frame_processor(display)
    
    return tracker_pipeline, track_event_pipeline


class PikaNodeExecutionFactory(PikaExecutionFactory):
    def __init__(self, db_conf: OmegaConf, show: bool) -> None:
        super().__init__()
        self.db_conf = db_conf
        self.conf_root:Path = Path("conf")
        self.show = show

    def create(self, pika_ctx: PikaExecutionContext) -> Execution:
        request = OmegaConf.create(pika_ctx.request)
        
        if exists_config(request, 'node'):
            path = self.conf_root / (request.node.replace(":", "_") + '.yaml')
            conf = load_config(path)
        elif exists_config(request, 'parameters'):
            conf = request.parameters
            conf.id = request.id
        else:
            raise ValueError(f'cannot get node configuration: request={request}')
        conf.show = self.show

        import json
        rtsp_uri = request.get('rtsp_uri', None)
        if rtsp_uri is None:
            raise ValueError(f'RTSP stream is not specified')
        rtsp_conf = OmegaConf.create({'uri': rtsp_uri})
        
        from dna.camera.utils import create_camera_from_conf
        camera = create_camera_from_conf(rtsp_conf)
        
        import dna
        img_proc = ImageProcessor(camera.open(), conf)
        build_node_processor(conf, image_processor=img_proc, context=pika_ctx)
        if dna.conf.exists_config(request, 'progress_report.interval_seconds'):
            interval = int(request.progress_report.interval_seconds)
            img_proc.report_interval = interval

        return img_proc