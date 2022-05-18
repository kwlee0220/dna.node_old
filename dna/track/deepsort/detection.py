# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    __slots__ = 'bbox', 'confidence', 'feature'
    
    def __init__(self, bbox, confidence, feature):
        self.bbox = bbox
        self.confidence = confidence
        self.feature = np.asarray(feature, dtype=np.float32)

    # def to_tlbr(self):
    #     return self.bbox.to_tlbr()

    # def to_xyah(self):
    #     return self.bbox.to_xyah()

    def __repr__(self) -> str:
        return f"({self.bbox},{self.confidence:.2f}"
