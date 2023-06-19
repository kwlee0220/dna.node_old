from __future__ import annotations

from typing import Optional
import logging

from omegaconf import OmegaConf

from dna import config
from dna.camera import ImageProcessor
from dna.track.track_pipeline import TrackingPipeline
from .track_event_pipeline import TrackEventPipeline
from .zone.zone_pipeline import ZonePipeline
from .zone.zone_sequences_display import ZoneSequenceDisplay
 

def build_node_processor(image_processor:ImageProcessor, conf: OmegaConf,
                         *,
                         tracking_pipeline:Optional[TrackingPipeline]=None) \
    -> tuple[ImageProcessor, TrackingPipeline, TrackEventPipeline]:
    # TrackingPipeline 생성하고 ImageProcessor에 등록함
    if not tracking_pipeline:
        tracker_conf = config.get_or_insert_empty(conf, 'tracker')
        tracking_pipeline = TrackingPipeline.load(tracker_conf)
    image_processor.add_frame_processor(tracking_pipeline)

    # TrackEventPipeline 생성하고 TrackingPipeline에 등록함
    publishing_conf = config.get_or_insert_empty(conf, 'publishing')
    logger = logging.getLogger("dna.node.event")
    track_event_pipeline = TrackEventPipeline(conf.id, publishing_conf=publishing_conf,
                                              image_processor=image_processor,
                                              logger=logger)
    tracking_pipeline.add_track_processor(track_event_pipeline)

    # ZonePipeline이 TrackEventPipeline에 등록되고, motion detection이 정의된 경우
    # 이를 ZonePipeline에 등록시킨다
    draw_motions = config.get(publishing_conf, "zone_pipeline.draw", default=True)
    zone_pipeline:ZonePipeline = track_event_pipeline.plugins.get('zone_pipeline')
    if zone_pipeline and image_processor.is_drawing and draw_motions:
        motion_detector = zone_pipeline.services.get('motions')
        motion_queue = zone_pipeline.event_queues.get('motions')
        if motion_detector and motion_queue:
            display = ZoneSequenceDisplay(motion_definitions=motion_detector.motion_definitions)
            motion_queue.add_listener(display)
            track_event_pipeline.add_listener(display)
            image_processor.add_frame_processor(display)
    
    return tracking_pipeline, track_event_pipeline
    