from __future__ import annotations

import sys
from typing import Tuple, Union, Dict, Any, Optional, List, TypeVar, Callable, Iterable
from datetime import datetime, timezone
from time import time
from pathlib import Path

from . import Box, Point
from .color import BGR

T = TypeVar("T")


def datetime2utc(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def utc2datetime(ts: int) -> datetime:
    return datetime.fromtimestamp(ts / 1000)

def datetime2str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

def utc_now() -> int:
    return round(time() * 1000)

def _parse_keyvalue(kv) -> Tuple[str,str]:
    pair = kv.split('=')
    if len(pair) == 2:
        return tuple(pair)
    else:
        return pair, None

def parse_query(query: str) -> Dict[str,str]:
    if not query or len(query) == 0:
        return dict()
    return dict([_parse_keyvalue(kv) for kv in query.split('&')])

def get_first_param(args: Dict[str,Any], key: str, def_value=None):
    value = args.get(key)
    return value[0] if value else def_value

def split_list(list:List, cond) -> Tuple[List,List]:
    trues = []
    falses = []
    for v in list:
        if cond(v):
            trues.append(v)
        else:
            falses.append(v)
    return trues, falses

from dna import color, plot_utils
import cv2
import numpy as np

def rindex(lst, value):
    return len(lst) - lst[::-1].index(value) - 1

def find_track_index(track_id, tracks):
    return next((idx for idx, track in enumerate(tracks) if track[idx].id == track_id), None)


def gdown_file(url:str, file: Path, force: bool=False):
    if isinstance(file, str):
        file = Path(file)
        
    if force:
        file.unlink()

    if not file.exists():
        # create an empty 'weights' folder if not exists
        file.parent.mkdir(parents=True, exist_ok=True)

        import gdown
        gdown.download(url, str(file.resolve().absolute()), quiet=False)

def initialize_logger(conf_file_path: Optional[str]=None):
    if conf_file_path is None:
        import pkg_resources
        conf_file_path = pkg_resources.resource_filename('conf', 'logger.yaml')
        
    with open(conf_file_path, 'rt') as f:
        import yaml
        import logging.config

        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        
        

def has_method(obj, name:str) -> bool:
    method = getattr(obj, name, None)
    return callable(method) if method else False
        
_RADIUS = 4
class RectangleDrawer:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.drawing = False
        self.coords = [0, 0, 0, 0]
        # self.bx, self.by, self.ex, self.ey = 0, 0, 0, 0

    def run(self) -> Tuple[np.ndarray, Box]:
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw)

        self.convas = self.image.copy()
        while ( True ):
            cv2.imshow('image', self.convas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyWindow('image')

        return self.convas, Box(self.coords)

    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.coords[0], self.coords[1] = x, y
            # self.bx, self.by = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            print(f'({x}),({y})')
            if self.drawing == True:
                self.convas = self.image.copy()
                cv2.rectangle(self.convas, tuple(self.coords[:2]), (x,y), (0,255,0), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.coords[2], self.coords[3] = x, y
            # self.ex, self.ey = x, y
            cv2.rectangle(self.convas, tuple(self.coords[:2]), (x,y), (0,255,0), 2)

    def is_on_corner(self, coord: List[float], radius=_RADIUS) -> int:
        pt = geometry.Point(coord).buffer(radius)
        for idx, coord in enumerate(self.coords):
            if geometry.Point(coord).intersects(pt):
                return idx
        return -1
    
    def is_on_the_line(self, coord: List[float], radius=_RADIUS) -> int:
        if len(self.coords) > 1:
            pt = geometry.Point(coord).buffer(radius)
            extended = self.coords + [self.coords[0]]
            for idx in list(range(len(extended)-1)):
                line = geometry.LineString([extended[idx], extended[idx+1]])
                if line.intersects(pt):
                    return idx
        return -1

from shapely import geometry
import dna.color as color
class PolygonDrawer:
    def __init__(self, image: np.ndarray, coords: List[List[float]]=[]) -> None:
        self.image = image
        self.drawing = False
        self.coords:List[List[float]] = coords
        self.index = -1
        self.selected_corner = -1
        self.selected_line = -1
        
        self.draw(-1, 0, 0, cv2.EVENT_LBUTTONUP, None)

    def run(self) -> List[List[float]]:
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw)

        self.convas = self.image.copy()
        while ( True ):
            cv2.imshow('image', self.convas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("output.png", self.convas)

        cv2.destroyWindow('image')
        return self.coords
        
    def is_on_corner(self, coord: List[float], radius=_RADIUS) -> int:
        pt = geometry.Point(coord).buffer(radius)
        for idx, coord in enumerate(self.coords):
            if geometry.Point(coord).intersects(pt):
                return idx
        return -1
    
    def is_on_line(self, coord: List[float], radius=_RADIUS) -> int:
        if len(self.coords) > 1:
            pt = geometry.Point(coord).buffer(radius)
            extended = self.coords + [self.coords[0]]
            for idx in list(range(len(extended)-1)):
                line = geometry.LineString([extended[idx], extended[idx+1]])
                if line.intersects(pt):
                    return idx
        return -1

    def draw(self, event, x, y, flags, param):
        cursor = [x,y]
        # print(f"coords={self.coords}, cursor={cursor}, index={self.index}, event={event}")
        n_coords = len(self.coords)
        if event == cv2.EVENT_LBUTTONDOWN:
            if n_coords == 0:
                self.coords = [cursor]
                self.index = 0
            elif n_coords == 1:
                self.coords.append(cursor)
                self.index = 0
            else:
                idx = self.is_on_corner(cursor)
                if idx >= 0:
                    del self.coords[idx]
                    self.coords.insert(idx, cursor)
                    self.index = idx
                else:
                    pt = geometry.Point(cursor).buffer(_RADIUS)
                    for idx in list(range(n_coords-1)):
                        line = geometry.LineString([self.coords[idx], self.coords[idx+1]])
                        if line.intersects(pt):
                            self.coords.insert(idx+1, cursor)
                            self.index = idx+1
                            break
                    if self.index < 0:
                        self.coords.append(cursor)
                        self.index = len(self.coords) - 1
        elif event == cv2.EVENT_LBUTTONUP:
            self.index = -1
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags == 1:
                if self.index >= 0:
                    del self.coords[self.index]
                    self.coords.insert(self.index, cursor)
            else:
                self.selected_corner = -1
                self.selected_line = -1
                idx = self.is_on_corner(cursor)
                if idx >= 0:
                    self.selected_corner = idx
                else:
                    idx = self.is_on_line(cursor)
                    if idx >= 0:
                        self.selected_line = idx 
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            idx = self.is_on_corner(cursor)
            if idx >= 0:
                del self.coords[idx]
        
        self.convas = self.image.copy()
        
        self.draw_polygon(self.coords, color.GREEN, 2)
        self.draw_corners(self.coords, color.RED, 3)
        
        if self.selected_line >= 0:
            self.highlight_polygon(self.coords, self.selected_line, color.GREEN, 5)
        if self.selected_corner >= 0:
            cv2.circle(self.convas, self.coords[self.selected_corner], 8, color.RED, -1)
        
    def _is_on_boundary(self, pt: Tuple[float,float]):
        return geometry.LineString(self.points).contains(geometry.Point(pt))
    
    def draw_polygon(self, coords:List[List[float]], color, thickness):
        if len(coords) > 2:
            coords = np.array(self.coords)
            cv2.polylines(self.convas, [coords], True, color, thickness, cv2.LINE_AA)
        elif len(coords) == 2:
            cv2.line(self.convas, coords[0], coords[1], color, thickness, cv2.LINE_AA)
        elif len(coords) == 1:
            cv2.circle(self.convas, coords[0], 3, color, -1)
            
    def draw_corners(self, coords:List[List[float]], color, radius):
        for coord in coords:
            cv2.circle(self.convas, coord, radius, color, -1)
            
    def highlight_polygon(self, coords:List[List[float]], index: int, color, thickness):
        n_coords = len(coords)
        if n_coords >= 2:
            if index < n_coords-1:
                cv2.line(self.convas, coords[index], coords[index+1], color, thickness, cv2.LINE_AA)
            else:
                cv2.line(self.convas, coords[index], coords[0], color, thickness, cv2.LINE_AA)
    
    def _to_int(self, pt: Tuple[float,float]):
        return round(pt[0]), round(pt[1])
    
    def _to_ints(self, pts: List[Tuple[float,float]]):
        return [self._to_int(pt) for pt in pts]