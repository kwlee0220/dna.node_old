import sys
from typing import Tuple, Union, Dict, Any, Optional
from datetime import datetime, timezone
from time import time
from pathlib import Path

from .types import Box
from .color import BGR


def datetime2utc(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def utc2datetime(ts: int) -> datetime:
    return datetime.fromtimestamp(ts / 1000)

def datetime2str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

def utc_now() -> int:
    return int(time() * 1000)

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


import logging
_LOGGERS = dict()
_LOG_FORMATTER = logging.Formatter("%(levelname)s: %(message)s (%(filename)s)")

def get_logger(name: str=None):
    logger = _LOGGERS.get(name)
    if not logger:
        logger = logging.getLogger(name)
        _LOGGERS[name] = logger
        
        logger.setLevel(logging.DEBUG)

        console = logging.StreamHandler()
        # console.setLevel(logging.INFO)
        console.setFormatter(_LOG_FORMATTER)
        logger.addHandler(console)
        
    return logger

from dna import color, plot_utils
import cv2
import numpy as np

def _draw_ds_track(convas, track, box_color:BGR, label_color:BGR, line_thickness:int):
    box = Box.from_tlbr(track.to_tlbr())
    box.draw(convas, box_color)
    if label_color:
        msg = f"{track.track_id}[{track.state}]"
        mat = plot_utils.draw_label(convas, msg, box.br.astype(int), label_color, box_color, 2)
    return convas

def draw_ds_tracks(convas, tracks, box_color, label_color=None, line_thickness=2, track_indices=None):
    if track_indices:
        tracks = [tracks[i] for i in track_indices]
    tracks = sorted(tracks, key=lambda t: t.track_id, reverse=True)

    for track in tracks:
        if track.is_tentative():
            convas = _draw_ds_track(convas, track, box_color, label_color, line_thickness)
    for track in tracks:
        if not track.is_tentative():
            convas = _draw_ds_track(convas, track, box_color, label_color, line_thickness)
    return convas

def draw_ds_detections(convas, dets, box_color, label_color=None, line_thickness=2):
    for idx, det in enumerate(dets):
        box = det.bbox
        box.draw(convas, box_color, line_thickness=line_thickness)
        if label_color:
            msg = f"{idx:02d}"
            mat = plot_utils.draw_label(convas, msg, box.br.astype(int), label_color, box_color, 2)
    return convas

def find_track_index(track_id, tracks):
    return next((idx for idx, track in enumerate(tracks) if track[idx].track_id == track_id), None)


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

class RectangleDrawer:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.drawing = False
        self.bx, self.by, self.ex, self.ey = 0, 0, 0, 0

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

        return self.convas, Box([self.bx, self.by, self.ex, self.ey])

    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.bx, self.by = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            print(f'({x}),({y})')
            if self.drawing == True:
                self.convas = self.image.copy()
                cv2.rectangle(self.convas, (self.bx, self.by), (x,y), (0,255,0), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.ex, self.ey = x, y
            cv2.rectangle(self.convas, (self.bx, self.by), (x,y), (0,255,0), 2)