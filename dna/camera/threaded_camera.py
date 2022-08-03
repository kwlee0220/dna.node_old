from __future__ import annotations
from typing import Optional
from enum import Enum
from threading import Condition, Thread
from pathlib import Path

from matplotlib.pyplot import show

from dna import Size2d
from .camera import Camera, ImageCapture
from .image_processor import ImageProcessor, FrameProcessor

class State(Enum):
    STOPPED = 1
    STARTING = 2
    RUNNING = 3
    STOPPING = 4

class ImageHolder(FrameProcessor):
    def __init__(self) -> None:
        self.cond = Condition()
        self.__state = State.STARTING
        self.__cap = None
        self.image = None

    @property
    def state(self) -> State:
        with self.cond:
            return self.__state

    @property
    def image_capture(self) -> Optional[ImageCapture]:
        with self.cond:
            return self.__cap

    @property
    def frame_index(self) -> int:
        with self.cond:
            return self.__cap.frame_index if self.__cap else -1

    def stop(self) -> None:
        with self.cond:
            if self.__state == State.RUNNING:
                self.__state = State.STOPPING
                self.cond.notifyAll()

    def on_started(self, proc: ImageProcessor) -> None:
        with self.cond:
            if self.__state == State.STARTING:
                self.__cap = proc.image_capture
                self.__state = State.RUNNING
                self.cond.notifyAll()
            else:
                raise AssertionError(f"unexpected state: {self.__state} for 'on_started'")

    def on_stopped(self) -> None:
        with self.cond:
            if self.__state != State.STOPPED:
                self.__state = State.STOPPED
                self.cond.notifyAll()

    def process_image(self, image: Image) -> Optional[Image]:
        with self.cond:
            if self.__state == State.RUNNING:
                self.image = image
                # print(f'\tupdated={image}')
                self.cond.notifyAll()
                return image
            elif self.__state == State.STOPPING:
                return None
            else:
                raise AssertionError(f"unexpected state: {self.__state} for 'process_image'")

    def await_image(self, min_frame_idx: int) -> Optional[Image]:
        with self.cond:
            while self.__state == State.STARTING or self.image is None:
                self.cond.wait()

            while self.__state == State.RUNNING:
                if self.image.frame_index < min_frame_idx:
                    self.cond.wait()
                else:
                    # print(f'awaited={self.image}')
                    return self.image

            return None

    def __repr__(self) -> str:
        frame_idx = self.image.frame_index if self.image else None
        return f"{self.__state}, frame={frame_idx}"


class ThreadedCamera(Camera):
    __slots__ = ('__base_cam', )

    def __init__(self, base_cam: Camera) -> None:
        self.__base_cam = base_cam

    @property
    def base_camera(self) -> Camera:
        return self.__base_cam

    @property
    def parameters(self) -> Camera.Parameters:
        return self.__base_cam.parameters

    def open(self) -> ThreadedImageCapture:
        return ThreadedImageCapture(self.__base_cam.open())


class ThreadedImageCapture(ImageCapture):
    __slots__ = '__holder', '__frame_idx', '__proc', 'thread'

    def __init__(self, capture: ImageCapture) -> None:
        self.__holder = ImageHolder()
        self.__frame_idx = 0
        self.__proc = ImageProcessor(ImageProcessor.Parameters(), capture, self.__holder)
        self.thread = Thread(target=self.__proc.run, args=tuple())
        self.thread.start()
        
    def close(self) -> None:
        self.__holder.stop()
        self.thread.join()

    def is_open(self) -> bool:
        return self.__holder.state == State.RUNNING

    def __call__(self) -> Optional[Image]:
        img: Image = self.__holder.await_image(self.__frame_idx+1)
        if img:
            self.__frame_idx = img.frame_index

        return img

    @property
    def size(self) -> Size2d:
        cap = self.__holder.image_capture
        return cap.size if cap else None

    @property
    def fps(self) -> int:
        cap = self.__holder.image_capture
        return cap.fps if cap else None

    @property
    def frame_index(self) -> int:
        return self.__holder.frame_index

    @property
    def repr_str(self) -> str:
        cap = self.__holder.image_capture
        return cap.repr_str if cap else None

    def __repr__(self) -> str:
        return f"{__class__.__name__}({self.repr_str})"