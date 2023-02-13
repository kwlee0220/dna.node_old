from omegaconf import OmegaConf


from dna import Box
_DEFAULT_DETECTOR = "dna.detect.yolov5:model=m&score=0.01"
def load(domain:Box, tracker_conf: OmegaConf):
    from dna.detect.utils import load_object_detector
    from ..dna_tracker import DNATracker

    detector_uri = tracker_conf.get("detector", _DEFAULT_DETECTOR)
    detector = load_object_detector(detector_uri)

    return DNATracker(detector, domain, tracker_conf)