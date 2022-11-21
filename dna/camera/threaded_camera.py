from __future__ import annotations
from typing import Optional
from enum import Enum
import time
from threading import Condition, Thread
from pathlib import Path
from omegaconf import OmegaConf

from matplotlib.pyplot import show

from dna import Size2d, color, Frame
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
        self.frame = None

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
                self.__cap = proc.capture
                self.__state = State.RUNNING
                self.cond.notifyAll()
            else:
                raise AssertionError(f"unexpected state: {self.__state} for 'on_started'")

    def on_stopped(self) -> None:
        with self.cond:
            if self.__state != State.STOPPED:
                self.__state = State.STOPPED
                self.cond.notifyAll()

    def process_frame(self, frame: Frame) -> Optional[Frame]:
        with self.cond:
            if self.__state == State.RUNNING:
                self.frame = frame
                self.cond.notifyAll()
                return frame
            elif self.__state == State.STOPPING:
                return None
            else:
                raise AssertionError(f"unexpected state: {self.__state} for 'process_frame'")

    def await_frame(self, min_frame_idx: int) -> Optional[Frame]:
        with self.cond:
            while self.__state == State.STARTING or self.frame is None:
                self.cond.wait()

            while self.__state == State.RUNNING:
                if self.frame.index < min_frame_idx:
                    self.cond.wait()
                else:
                    return self.frame

            return None

    def __repr__(self) -> str:
        frame_idx = self.frame.index if self.frame else None
        return f"{self.__state}, frame={frame_idx}"


class ThreadedCamera(Camera):
    __slots__ = ('__base_cam', )

    def __init__(self, base_cam: Camera) -> None:
        self.__base_cam = base_cam

    @property
    def base_camera(self) -> Camera:
        return self.__base_cam

    def open(self) -> ThreadedImageCapture:
        return ThreadedImageCapture(self.__base_cam.open())
        
    @property
    def uri(self) -> str:
        return self.__base_cam.uri

    def size(self) -> Size2d:
        return self.__base_cam.size()

class ThreadedImageCapture(ImageCapture):
    __slots__ = '__holder', '__frame_idx', '__proc', 'thread'

    def __init__(self, capture: ImageCapture) -> None:
        self.__holder = ImageHolder()
        self.__frame_idx = 0

        conf = OmegaConf.create()
        self.__proc = ImageProcessor(capture, conf)
        self.__proc.add_frame_processor(self.__holder)
        self.thread = Thread(target=self.__proc.run, args=tuple())
        self.thread.start()
        self.__proc.wait_for_ready()
        
    def close(self) -> None:
        self.__proc.close()
        self.thread.join()

    def is_open(self) -> bool:
        return self.__holder.state == State.RUNNING

    def __call__(self) -> Optional[Frame]:
        frame: Frame = self.__holder.await_frame(self.__frame_idx+1)
        if frame:
            self.__frame_idx = frame.index

        return frame

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