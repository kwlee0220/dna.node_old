from __future__ import annotations
import imp

from typing import List

from omegaconf import OmegaConf

from dna import Box
from dna.camera import Camera, ImageProcessor
from .track_callbacks import TrackerCallback, ObjectTrackingCallback


def load_object_tracking_callback(camera: Camera, proc: ImageProcessor, conf: OmegaConf,
                                tracker_callbacks: List[TrackerCallback]=[]) -> ObjectTrackingCallback:
    from dna.detect.utils import load_object_detector
    from .track_callbacks import TrackWriter, ObjectTrackingCallback
    from .deepsort_tracker import DeepSORTTracker

    domain = Box.from_size(camera.size)
    draw_tracks = proc.is_drawing()

    output = conf.get("output", None)
    show_zones = conf.get('show_zones', False)

    detector = load_object_detector(conf.detector)
    tracker = DeepSORTTracker(detector, domain, conf)
    tracker_cbs = [TrackWriter(output)] + tracker_callbacks if output else tracker_callbacks
    return ObjectTrackingCallback(tracker, tracker_cbs, draw_tracks, show_zones)