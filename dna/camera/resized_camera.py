from __future__ import annotations
from typing import Optional

import cv2

from dna import Size2d
from .camera import Camera, ImageCapture, Image


class ResizingCamera(Camera):
    def __init__(self, camera: Camera, target_size: Size2d):
        self.base_camera = camera
        self.__size = target_size

    def open(self) -> ImageCapture:
        src_capture = self.base_camera.open()
        return ResizingImageCapture(self, src_capture) if src_capture.size != self.__size else src_capture

    @property
    def size(self) -> Size2d:
        return self.__size

    def __repr__(self) -> str:
        end_frame = self.end_frame if self.end_frame else ""
        return f"Range({self.base_camera}, {self.begin_frame}:{end_frame})"


class ResizingImageCapture(ImageCapture):
    def __init__(self, camera: ResizingCamera, capture: ImageCapture) -> None:
        self.__cap = capture
        self.__size = camera.size

        src_size = capture.size
        if self.__size == src_size:
            raise ValueError(f"target size({self.__size}) is equal to the source size({src_size})")
        elif self.__size.area() < src_size.area():
            self.interpolation = cv2.INTER_AREA
        elif self.__size.area() > src_size.area():
            self.interpolation = cv2.INTER_LINEAR
        else:
            raise ValueError(f"invalid target size: {self.__size}")

    def close(self) -> None:
        self.__cap.close()

    def is_open(self) -> bool:
        return self.__cap.is_open()

    def __call__(self) -> Optional[Image]:
        img = self.__cap()
        if img:
            mat = cv2.resize(img.mat, dsize=self.__size.to_tuple(), interpolation=self.interpolation)
            img = Image(mat=mat, frame_index=img.frame_index, ts=img.ts)
        return img

    @property
    def size(self) -> Size2d:
        return self.__size

    @property
    def fps(self) -> int:
        return self.__cap.fps

    @property
    def frame_index(self) -> int:
        return self.__cap.frame_index

    @property
    def repr_str(self) -> str:
        return f'{self.__cap.repr_str}, target_size={self.__size}'

    def __repr__(self) -> str:
        return f'{self.__camera}({self.repr_str})'