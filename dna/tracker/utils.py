from __future__ import annotations
from typing import List

from omegaconf import OmegaConf

from dna import Box
from dna.camera import Camera, ImageProcessor
from .tracker import ObjectTracker
from .track_callbacks import TrackerCallback, ObjectTrackingCallback


def load_object_tracking_callback(camera: Camera, proc: ImageProcessor, tracker_conf: OmegaConf,
                                tracker_callbacks: List[TrackerCallback]=[]) -> ObjectTrackingCallback:
    from dna.detect.utils import load_object_detector
    from .track_callbacks import TrackWriter, ObjectTrackingCallback

    domain = Box.from_size(camera.size())
    tracker_uri = tracker_conf.get("uri", "dna.tracker.dna_deepsort")
    tracker = load_tracker(tracker_uri, domain, tracker_conf)

    draw_tracks = proc.is_drawing() and tracker_conf.get("draw_tracks", proc.is_drawing())
    draw_zones = tracker_conf.get("draw_zones", False)
    output = tracker_conf.get("output", None)

    tracker_cbs = [TrackWriter(output)] + tracker_callbacks if output else tracker_callbacks
    return ObjectTrackingCallback(tracker, tracker_cbs, draw_tracks, draw_zones)


def load_tracker(uri: str, domain: Box, conf: OmegaConf) -> ObjectTracker:
    parts = uri.split(':', 1)
    id, query = tuple(parts) if len(parts) > 1 else (uri, "")

    import importlib
    tracker_module = importlib.import_module(id)
    return tracker_module.load(domain, conf)
