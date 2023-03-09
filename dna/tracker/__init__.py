from omegaconf import OmegaConf

from .type import TrackState, ObjectTrack, ObjectTracker, TrackProcessor
from .track_pipeline import TrackingPipeline


_DEFAULT_DETECTOR = "dna.detect.yolov5:model=l6&score=0.01&agnostic=True&max_det=50&classes=car,bus,truck"
# _DEFAULT_DETECTOR = "dna.detect.ultralytics:model=yolov8l&type=v8&score=0.1&classes=car,bus,truck&agnostic_nms=True"
def load_dna_tracker(tracker_conf: OmegaConf):
    from dna.detect.utils import load_object_detector
    from .dna_tracker import DNATracker

    detector_uri = tracker_conf.get("detector", _DEFAULT_DETECTOR)
    detector = load_object_detector(detector_uri)

    return DNATracker(detector, tracker_conf)