# vim: expandtab:ts=4:sw=4
import numpy as np

from dna import Box


class Detection(object):
    __slots__ = 'bbox', 'score', 'feature'
    
    def __init__(self, bbox:Box, score:float, feature):
        self.bbox = bbox
        self.score = score
        self.feature = np.asarray(feature, dtype=np.float32)

    # def to_tlbr(self):
    #     return self.bbox.tlbr

    # def to_xyah(self):
    #     return self.bbox.to_xyah()

    def __repr__(self) -> str:
        return f"({self.bbox},{self.score:.2f}"
