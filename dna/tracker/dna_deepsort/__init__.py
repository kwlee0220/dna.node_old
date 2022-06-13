from omegaconf import OmegaConf

from dna import Box
from dna.detect.utils import load_object_detector
from .deepsort_tracker import DeepSORTTracker

def load(domain:Box, tracker_conf: OmegaConf):
    impl_conf = tracker_conf.get('dna_deepsort', None)
    detector_uri = impl_conf.get("detector")
    detector = load_object_detector(detector_uri)
    return DeepSORTTracker(detector, domain, impl_conf)