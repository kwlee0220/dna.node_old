from typing import Optional, List

from omegaconf import OmegaConf
from pathlib import Path

from dna import Box
from .object_detector import ObjectDetector, ScoreFilteredObjectDetector, \
                            LabelFilteredObjectDetector, BlindZoneObjectDetector


def load_object_detector(uri: str) -> ObjectDetector:
    if not uri:
        raise ValueError(f"detector id is None")

    parts = uri.split(':', 1)
    id, query = tuple(parts) if len(parts) > 1 else (uri, "")
    if id == 'file':
        from pathlib import Path
        from .object_detector import LogReadingDetector

        det_file = Path(query)
        return LogReadingDetector(det_file)
    else:
        import importlib
        
        loader_module = importlib.import_module(id)
        return loader_module.load(query)


def filter_by_score(detector:ObjectDetector, min_score:float) -> ObjectDetector:
    return ScoreFilteredObjectDetector(detector, min_score)

def filter_by_labels(detector:ObjectDetector, accept_labels:List[str]) -> ObjectDetector:
    return LabelFilteredObjectDetector(detector, accept_labels)

def filter_by_blind_zones(detector:ObjectDetector, blind_zones:List[Box]) -> ObjectDetector:
    return BlindZoneObjectDetector(detector, blind_zones)

from .detecting_callback import DetectingCallback
def load_object_detecting_callback(detector_uri:str, output: Optional[Path]=None,
                                    draw_detections: bool=False) -> DetectingCallback:
    detector = load_object_detector(detector_uri)
    return DetectingCallback(detector=detector, output=output, draw_detections=draw_detections)