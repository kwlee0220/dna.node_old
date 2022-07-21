from omegaconf import OmegaConf

from dna import Box
from dna.detect.utils import load_object_detector
from .deepsort_tracker import DeepSORTTracker


_DEFAULT_DETECTOR = "dna.detect.yolov5:model=m&score=0.45"

def load(domain:Box, tracker_conf: OmegaConf):
    impl_conf = tracker_conf.get('dna_deepsort', OmegaConf.create())
    detector_uri = impl_conf.get("detector", _DEFAULT_DETECTOR)
    detector = load_object_detector(detector_uri)
    return DeepSORTTracker(detector, domain, impl_conf)