from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NewType, Tuple, Optional, Union
import math

import numpy as np
import cv2

from .color import BGR


class Point:
    __slots__ = ('__xy', )

    def __init__(self, x: Union[int, float], y: Union[int, float]) -> None:
        self.__xy = np.array([x, y])

    @property
    def x(self) -> Union[int, float]:
        return self.__xy[0]

    @property
    def y(self) -> Union[int, float]:
        return self.__xy[1]

    @property
    def xy(self):
        return self.__xy

    def to_tuple(self) -> Tuple[Union[int, float], Union[int, float]]:
        return tuple(self.__xy)

    @classmethod
    def from_np(cls, xy: np.ndarray) -> Point:
        return Point(xy[0], xy[1])

    def distance_to(self, pt:Point) -> float:
        return np.linalg.norm(self.xy - pt.xy)

    def angle_with(self, pt:Point) -> float:
        delta = pt - self
        return math.atan2(delta.height, delta.width)

    @staticmethod
    def line_function(pt1:Point, pt2:Point):
        delta = pt1.xy - pt2.xy
        if delta[0] == 0:
            raise ValueError(f"Cannot find a line function: {pt1} - {pt2}")
        slope = delta[1] / delta[0]
        y_int = pt2.y - (slope * pt2.x)

        def func(x):
            return (slope * x) + y_int
        return func

    @staticmethod
    def split_points(pt1:Point, pt2:Point, npoints:int) -> List[Point]:
        func = Point.line_function(pt1, pt2)
        step_x = (pt2.x - pt1.x) / (npoints+1)
        xs = [pt1.x + (idx * step_x) for idx in range(1, npoints+1)]
        return [Point.from_np(np.array([x, func(x)])) for x in xs]

    def to_rint(self) -> Point:
        return Point.from_np(np.rint(self.__xy).astype(int))

    def __add__(self, rhs) -> Point:
        if isinstance(rhs, Point):
            return Point.from_np(self.xy + rhs.xy)
        elif isinstance(rhs, Size2d):
            return Point.from_np(self.xy + rhs.wh)
        elif (isinstance(rhs, np.ndarray) or isinstance(rhs, tuple) or isinstance(rhs, list)) and len(rhs) >= 2:
            return Point(self.x + rhs[0], self.y + rhs[1])
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Point(self.x + rhs, self.y + rhs)
        else:
            raise ValueError(f"invalid rhs: rhs={rhs}")

    def __sub__(self, rhs) -> Union[Point,Size2d]:
        if isinstance(rhs, Point):
            return Size2d.from_np(self.xy - rhs.xy)
        elif isinstance(rhs, Size2d):
            return Point.from_np(self.xy - rhs.wh)
        elif isinstance(rhs, tuple) and len(rhs) >= 2:
            return Point(self.x - rhs[0], self.y - rhs[1])
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Point(self.x - rhs, self.y - rhs)
        else:
            raise ValueError(f"invalid rhs: rhs={rhs}")

    def __mul__(self, rhs) -> Point:
        if isinstance(rhs, int) or isinstance(rhs, float):
            return Point(self.x * rhs, self.y * rhs)
        elif isinstance(rhs, Size2d):
            return Point.from_np(self.xy * rhs.wh)
        elif isinstance(rhs, tuple) and len(rhs) >= 2:
            return Point(self.x * rhs[0], self.y * rhs[1])
        else:
            raise ValueError(f"invalid rhs: rhs={rhs}")

    def __truediv__(self, rhs) -> Point:
        if isinstance(rhs, Size2d):
            return Point(self.x / rhs.width, self.y / rhs.height)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Point(self.x / rhs, self.y / rhs)
        else:
            raise ValueError('invalid right-hand-side:', rhs)
    
    def __repr__(self) -> str:
        if isinstance(self.xy[0], np.int32):
            return '({},{})'.format(*self.xy)
        else:
            return '({:.1f},{:.1f})'.format(*self.xy)

class Size2d:
    __slots__ = ('__wh',)

    def __init__(self, width: Union[int, float], height: Union[int, float]) -> None:
        self.__wh = np.array([width, height])

    @classmethod
    def from_np(cls, wh:np.ndarray) -> Size2d:
        return Size2d(wh[0], wh[1])

    @classmethod
    def parse_string(cls, expr:str) -> Size2d:
        parts: List[int] = [int(p) for p in expr.split("x")]
        if len(parts) == 2:
            return Size2d(parts[0], parts[1])
        raise ValueError(f"invalid Size2d string: {expr}")

    @staticmethod
    def from_conf(expr:object) -> Size2d:
        if isinstance(expr, str):
            return Size2d.parse_string(expr)
        elif isinstance(expr, Size2d):
            return expr
        elif hasattr(expr, "__getitem__"):
            return Size2d(expr[0], expr[1])
        else:
            raise ValueError(f'invalid Size2d expression: {expr}')

    def to_tuple(self) -> Tuple[Union[int, float],Union[int, float]]:
        return tuple(np.rint(self.wh).astype(int))
        # return tuple(self.__wh)

    def is_valid(self) -> bool:
        return self.__wh[0] >= 0 and self.__wh[1] >= 0

    @property
    def wh(self) -> np.ndarray:
        return self.__wh

    @property
    def width(self) -> Union[int, float]:
        return self.__wh[0]
    
    @property
    def height(self) -> Union[int, float]:
        return self.__wh[1]

    def aspect_ratio(self) -> float:
        return self.__wh[0] / self.__wh[1]

    def area(self) -> float:
        return self.__wh[0] * self.__wh[1]

    def abs(self) -> Size2d:
        return Size2d.from_np(np.abs(self.__wh))

    def to_rint(self) -> Size2d:
        return Size2d.from_np(np.rint(self.wh).astype(int))

    def norm(self):
        return np.linalg.norm(self.wh)

    def __add__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d):
            return Size2d.from_np(self.wh + rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d.from_np(self.wh - np.array([rhs, rhs]))
        else:
            raise ValueError('invalid right-hand-side:', rhs)

    def __sub__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d):
            return Size2d.from_np(self.wh - rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d.from_np(self.wh - np.array([rhs, rhs]))
        else:
            raise ValueError('invalid right-hand-side:', rhs)

    def __mul__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d):
            return Size2d.from_np(self.wh * rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d.from_np(self.wh * np.array([rhs, rhs]))
        else:
            raise ValueError('invalid right-hand-side:', rhs)

    def __truediv__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d):
            return Size2d.from_np(self.wh / rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d.from_np(self.wh / np.array([rhs, rhs]))
        else:
            raise ValueError('invalid right-hand-side:', rhs)

    def __eq__(self, other):
        if isinstance(other, Size2d):
            return np.array_equal(self.__wh, other.__wh)
        else:
            return False
    
    def __repr__(self) -> str:
        if isinstance(self.wh[0], np.int32):
            return '{}x{}'.format(*self.__wh)
        else:
            return '{:.1f}x{:.1f}'.format(*self.__wh)
EMPTY_SIZE2D: Size2d = Size2d(-1, -1)


class Box:
    __slots__ = ('__tlbr', )

    def __init__(self, tlbr: np.ndarray) -> None:
        self.__tlbr = np.array(tlbr)

    @classmethod
    def from_points(self, tl: Point, br: Point) -> Box:
        return Box(np.hstack([tl.xy, br.xy]))

    @classmethod
    def from_tlbr(cls, tlbr: np.ndarray) -> Box:
        return Box(tlbr)

    @classmethod
    def from_tlwh(cls, tlwh: np.ndarray) -> Box:
        tlbr = tlwh.copy()
        tlbr[2:] = tlwh[:2] + tlwh[2:]
        return Box(tlbr)

    @staticmethod
    def from_size(size:Size2d) -> Box:
        return Box.from_tlbr(np.array([0, 0, size.width, size.height]))

    def translate(self, x: Union[int, float], y: Union[int, float]) -> Box:
        tl = self.tl
        wh = self.wh
        return Box.from_tlwh(np.array([tl[0]+x, tl[1]+y, wh[0], wh[1]]))

    def is_valid(self) -> bool:
        wh = self.wh
        return wh[0] >= 0 and wh[1] >= 0

    def to_tlbr(self) -> np.ndarray:
        return self.__tlbr

    def to_tlwh(self) -> np.ndarray:
        tlwh = self.__tlbr.copy()
        tlwh[2:] = self.br - self.tl
        return tlwh

    def to_xyah(self) -> np.ndarray:
        ret = self.to_tlwh()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_rint(self) -> Box:
        return Box(np.rint(self.__tlbr).astype(int))

    @property
    def tl(self) -> np.ndarray:
        return self.__tlbr[:2]

    @property
    def br(self) -> np.ndarray:
        return self.__tlbr[2:]

    @property
    def wh(self) -> np.ndarray:
        return self.br - self.tl

    @property
    def width(self) -> Union[float,int]:
        return self.wh[0]

    @property
    def height(self) -> Union[float,int]:
        return self.wh[1]
    
    @property
    def coords(self) -> np.ndarray:
        return np.array([[self.__tlbr[0], self.__tlbr[1]],
                            [self.__tlbr[2], self.__tlbr[1]],
                            [self.__tlbr[2], self.__tlbr[3]],
                            [self.__tlbr[0], self.__tlbr[3]]])

    def top_left(self) -> Point:
        return Point.from_np(self.tl)

    def bottom_right(self) -> Point:
        return Point.from_np(self.br)

    def center(self) -> Point:
        return Point.from_np(self.tl + (self.wh / 2.))

    def size(self) -> Size2d:
        return Size2d.from_np(self.wh) if self.is_valid() else EMPTY_SIZE2D

    def aspect_ratio(self) -> float:
        return self.wh[0] / self.wh[1]

    def area(self) -> float:
        return self.size().area() if self.is_valid() else 0

    def area_int(self) -> int:
        if self.is_valid():
            wh = self.wh + 1
            return wh[0] * wh[1]
        else:
            return 0

    def distance_to(self, box:Box) -> float:
        tlbr1 = self.__tlbr
        tlbr2 = box.__tlbr

        delta1 = tlbr1[[0,3]] - tlbr2[[2,1]]
        delta2 = tlbr2[[0,3]] - tlbr2[[2,1]]
        u = np.max(np.array([np.zeros(len(delta1)), delta1]), axis=0)
        v = np.max(np.array([np.zeros(len(delta2)), delta2]), axis=0)
        dist = np.linalg.norm(np.concatenate([u, v]))
        return dist

    def contains(self,box:Box) -> bool:
        return self.__tlbr[0] <= box.__tlbr[0] and self.__tlbr[1] <= box.__tlbr[1] \
                and self.__tlbr[2] >= box.__tlbr[2] and self.__tlbr[3] >= box.__tlbr[3]

    def intersection(self, bbox:Box) -> Optional[Box]:
        x1 = max(self.__tlbr[0], bbox.__tlbr[0])
        y1 = max(self.__tlbr[1], bbox.__tlbr[1])
        x2 = min(self.__tlbr[2], bbox.__tlbr[2])
        y2 = min(self.__tlbr[3], bbox.__tlbr[3])
        
        if x1 >= x2 or y1 >= y2:
            return EMPTY_BOX
        else:
            return Box.from_tlbr(np.array([x1, y1, x2, y2]))

    def iou(self,box:Box) -> float:
        inter_area = self.intersection(box).area()
        area1, area2 = self.area(), self.area()
        return inter_area / (area1 + area2 - inter_area)

    def iou_int(self,box:Box) -> float:
        inter_area = self.intersection(box).area_int()
        area1, area2 = self.area_int(), self.area_int()
        return inter_area / (area1 + area2 - inter_area)

    def overlap_ratios(self, other:Box) -> Tuple[float,float,float]:
        inter_area = self.intersection(other).area()
        r1 = inter_area / self.area() if self.is_valid() else 0
        r2 = inter_area / other.area() if other.is_valid() else 0
        iou = inter_area / (self.area() + other.area() - inter_area)  if self.is_valid() and other.is_valid() else 0
        return (r1, r2, iou)

    def draw(self, convas:Image, color:BGR, line_thickness=2):
        tlbr_int = self.__tlbr.astype(int)
        return cv2.rectangle(convas, tlbr_int[0:2], tlbr_int[2:4], color,
                            thickness=line_thickness, lineType=cv2.LINE_AA)

    def crop(self, img:Image) -> np.ndarray:
        tl = self.tl
        br = self.br
        return img[tl[1]:br[1], tl[0]:br[0]]

    def copy(self, src_img:Image, tar_img:Image) -> None:
        x, y = tuple(self.tl)
        w, h = tuple(self.wh)
        tar_img[y:y+h, x:x+w] = src_img
    
    def __repr__(self):
        return '{}:{}'.format(self.top_left(), self.size())

EMPTY_BOX:Box = Box(np.array([-1,-1,0,0]))


Image = NewType('Image', np.ndarray)
@dataclass(frozen=True, eq=True)    # slots=True
class Frame:
    image: Image = field(repr=False)
    index: int
    ts: float
    
    def copy(self) -> Frame:
        return Frame(self.image.copy(), self.index, self.ts)