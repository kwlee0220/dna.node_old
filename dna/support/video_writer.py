from __future__ import annotations
from typing import Optional, Tuple, List

from pathlib import Path
from contextlib import suppress
import logging

import cv2

from dna import Image, Box

class VideoWriter:
    def __init__(self, video_file: str, fps:int, size:Tuple[int,int]) -> None:
        super().__init__()
        
        path = Path(video_file)

        self.fourcc = None
        ext = path.suffix.lower()
        if ext == '.mp4':
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif ext == '.avi':
            self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        else:
            raise IOError("unknown output video file extension: 'f{ext}'")
        self.path = path.resolve()
        
        self.fps = fps
        self.size = size
        
    def __enter__(self):
        self.path.parent.mkdir(exist_ok=True)
        self.video_writer = cv2.VideoWriter(str(self.path), self.fourcc, self.fps, self.size)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        with suppress(Exception): self.video_writer.release()

    def write(self, image:Image) -> None:
        self.video_writer.write(image)