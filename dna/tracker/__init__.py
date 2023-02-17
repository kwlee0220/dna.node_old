from omegaconf import OmegaConf

from .type import TrackState, ObjectTrack, ObjectTracker, TrackProcessor
from .track_pipeline import TrackingPipeline


from dna import Box
_DEFAULT_DETECTOR = "dna.detect.yolov5:model=l6&score=0.01"
def load_dna_tracker(tracker_conf: OmegaConf):
    from dna.detect.utils import load_object_detector
    from .dna_tracker import DNATracker

    detector_uri = tracker_conf.get("detector", _DEFAULT_DETECTOR)
    detector = load_object_detector(detector_uri)

    return DNATracker(detector, tracker_conf)