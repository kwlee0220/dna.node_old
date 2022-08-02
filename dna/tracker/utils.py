from __future__ import annotations
from typing import List

from omegaconf import OmegaConf

from dna import Box
from dna.camera import ImageProcessor
from .tracker import ObjectTracker
from .track_pipeline import TrackProcessor, ObjectTrackPipeline, TrackWriter


def build_track_pipeline(proc: ImageProcessor, tracker_conf: OmegaConf,
                         track_processors: List[TrackProcessor]=[]) -> ObjectTrackPipeline:
    domain = Box.from_size(proc.capture.size)
    tracker_uri = tracker_conf.get("uri", "dna.tracker.dna_deepsort")
    tracker = load_tracker(tracker_uri, domain, tracker_conf)

    draw_tracks = proc.is_drawing() and tracker_conf.get("draw_tracks", proc.is_drawing())
    draw_zones = tracker_conf.get("draw_zones", False)

    output = tracker_conf.get("output", None)
    if output is not None:
        track_processors = [TrackWriter(output)] + track_processors
        
    return ObjectTrackPipeline(tracker, track_processors, draw_tracks, draw_zones)


def load_tracker(uri: str, domain: Box, conf: OmegaConf) -> ObjectTracker:
    parts = uri.split(':', 1)
    id, query = tuple(parts) if len(parts) > 1 else (uri, "")

    import importlib
    tracker_module = importlib.import_module(id)
    return tracker_module.load(domain, conf)
