from __future__ import annotations
from typing import Optional
import time

import numpy as np

from dna import Size2d
from .camera import Camera, ImageCapture, Image
from .opencv_camera import OpenCvImageCapture


class RangedCamera(Camera):
    __slots__ = 'source_camera', 'begin_frame', 'end_frame'

    def __init__(self, camera: Camera, begin_frame: int=1, end_frame: Optional[int]=None):
        self.source_camera = camera
        self.begin_frame = begin_frame
        self.end_frame = end_frame

    def open(self) -> ImageCapture:
        src_capture = self.source_camera.open()
        return RangedImageCapture(self, src_capture)

    def size(self) -> Optional[Size2d]:
        return self.source_camera.size()

    def __repr__(self) -> str:
        end_frame = self.end_frame if self.end_frame else ""
        return f"Range({self.source_camera}, {self.begin_frame}:{end_frame})"


class RangedImageCapture(ImageCapture):
    __slots__ = '__camera', '__cap'

    def __init__(self, camera: RangedCamera, capture: ImageCapture) -> None:
        self.__camera = camera
        self.__cap = capture

        if isinstance(self.__cap, OpenCvImageCapture):
            self.__cap.set_frame_index(camera.begin_frame)
        else:
            # ignore images until the frame_index becomes begin_frame.
            while self.__cap.frame_index < camera.begin_frame:
                img: Image = self.__cap()
                if img is None:
                    break

    def close(self) -> None:
        self.__cap.close()

    def is_open(self) -> bool:
        return self.__cap.is_open()

    def __call__(self) -> Optional[Image]:
        if self.__camera.end_frame and self.frame_index >= self.__camera.end_frame:
            return None

        return self.__cap()

    @property
    def size(self) -> Size2d:
        return self.__cap.size

    @property
    def fps(self) -> int:
        return self.__cap.fps

    @property
    def sync(self) -> bool:
        return self.__cap.sync

    @sync.setter
    def sync(self, flag) -> None:
        return self.__cap.sync(flag)

    @property
    def frame_index(self) -> int:
        return self.__cap.frame_index

    @property
    def total_frame_count(self) -> int:
        if hasattr(self.__cap, 'total_frame_count'):
            return self.__cap.total_frame_count
        else:
            return self.__camera.end_frame - self.__camera.begin_frame + 1

    @property
    def repr_str(self) -> str:
        end_frame = self.__camera.end_frame if self.__camera.end_frame else ""
        return f'{self.__cap.repr_str}, range=[{self.__camera.begin_frame}:{end_frame}]'

    def __repr__(self) -> str:
        return f'{self.__camera}({self.repr_str})'