from __future__ import annotations
from typing import List, Optional

import numpy as np

from dna import BGR, Box, Size2d, Image
from dna.utils import plot_utils


class Detection:
    __slots__ = 'bbox', 'label', 'score', 'feature'

    def __init__(self, bbox:Box, label:Optional[str]=None, score:float=-1) -> None:
        self.bbox = bbox
        self.label = label
        self.score = score
        self.feature = None

    def draw(self, convas: Image, color:BGR, label_color:Optional[BGR]=None, show_score:bool=True,
            line_thickness:int=2) -> Image:
        loc = self.bbox
        convas = loc.draw(convas, color=color, line_thickness=line_thickness)
        if label_color:
            msg = f"{self.label}({self.score:.3f})" if show_score else self.label
            convas = plot_utils.draw_label(convas=convas, label=msg, tl=loc.tl.astype(int),
                                            color=label_color, fill_color=color, thickness=2)

        return convas

    def __truediv__(self, rhs) -> Detection:
        if isinstance(rhs, Size2d):
            return Detection(bbox=self.bbox/rhs, label=self.label, score=self.score)
        else:
            raise ValueError('invalid right-hand-side:', rhs)
    
    def __repr__(self) -> str:
        return f'{self.label}:{self.bbox},{self.score:.3f}'