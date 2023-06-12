from __future__ import annotations
from typing import Optional

from pathlib import Path
from contextlib import suppress

import cv2

from dna import Image, Size2d


class VideoWriter:
    __slots__ = ('fourcc', 'path', 'fps', 'size', 'video_writer')
    
    def __init__(self, video_file:str, fps:int, size:Size2d) -> None:
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
        self.video_writer = None
        
    def open(self) -> None:
        assert not self.is_open(), "already opened."
        
        self.path.parent.mkdir(exist_ok=True)
        self.video_writer = cv2.VideoWriter(str(self.path), self.fourcc, self.fps, tuple(self.size.to_rint().wh))
        
    def close(self) -> None:
        if self.is_open():
            self.video_writer.release()
            self.video_writer = None
        
    def is_open(self) -> bool:
        return self.video_writer is not None
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        with suppress(Exception): self.close()

    def write(self, image:Image) -> None:
        assert self.is_open(), "not opened."
        self.video_writer.write(image)