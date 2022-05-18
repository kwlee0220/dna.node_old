from typing import List, Optional

import numpy as np
import cv2

from dna import Box, Point, BGR, WHITE, Image


def draw_line(convas:Image, from_pt:Point, to_pt:Point, color:BGR,
                line_thickness: int=2) -> Image:
    return draw_line_raw(convas, from_pt.xy.astype(int), to_pt.xy.astype(int), color, line_thickness)

def draw_line_raw(convas:Image, from_pt, to_pt, color:BGR, line_thickness: int=2) -> Image:
    return cv2.line(convas, from_pt, to_pt, color, line_thickness)

def draw_line_string_raw(convas:Image, pts:List[List[int]], color: BGR,
                            line_thickness: int=2) -> Image:
    for pt1, pt2 in zip(pts, pts[1:]):
        convas = draw_line_raw(convas, pt1, pt2, color, line_thickness)
    return convas

def draw_line_string(convas:Image, pts: List[Point], color:BGR, line_thickness: int=2) -> Image:
    return draw_line_string_raw(convas, [pt.xy.astype(int) for pt in pts], color, line_thickness)

def draw_label(convas:Image, label:str, tl, color: BGR=WHITE, fill_color: Optional[BGR]=None,
                thickness: int=2) -> Image:
    txt_thickness = max(thickness - 1, 1)
    scale = thickness / 4

    txt_size = cv2.getTextSize(label, 0, fontScale=scale, thickness=thickness)[0]
    br = (tl[0] + txt_size[0], tl[1] - txt_size[1] - 3)
    convas = cv2.rectangle(convas, tl, br, fill_color, -1, cv2.LINE_AA)  # filled
    return cv2.putText(convas, label, (tl[0], tl[1] - 2), 0, scale, color, thickness=txt_thickness,
                        lineType=cv2.LINE_AA)