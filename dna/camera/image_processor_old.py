from __future__ import annotations

from dataclasses import dataclass, Field
from contextlib import closing
from typing import Optional, Tuple
from pathlib import Path
from abc import ABCMeta, abstractmethod
import time

from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from dna import color, Frame
from .camera import Camera, ImageCapture
from dna.execution import Execution, ExecutionContext, NoOpExecutionContext


class ImageProcessorCallback(metaclass=ABCMeta):
    @abstractmethod
    def on_started(self, proc:ImageProcessor) -> None:
        pass

    @abstractmethod
    def on_stopped(self) -> None:
        pass

    @abstractmethod
    def process_image(self, frame:Frame) -> Optional[Frame]:
        pass

    def set_control(self, key:int) -> int:
        return key


class DefaultCallback(ImageProcessorCallback):
    def on_started(self, proc:ImageProcessor) -> None: pass
    def on_stopped(self) -> None: pass
    def process_image(self, image:Frame) -> Optional[Frame]: return image


class ImageProcessor(Execution):
    __ALPHA = 0.2
    __DEFAULT_CALLBACK = DefaultCallback()
    __slots__ = ('__camera', '__cb', 'window_name', 'output_video', 'show_progress',
                'pause_on_eos', 'writer', '__fps_measured')

    def __init__(self, camera: Camera, conf: OmegaConf, context: ExecutionContext=NoOpExecutionContext()):
        self.__camera = camera
        self.__cb = ImageProcessor.__DEFAULT_CALLBACK
        self.ctx = context

        self.window_name:str = conf.get("window_name", None)
        self.output_video:str = conf.get("output_video", None)
        self.show_progress:bool = conf.get("show_progress", False)
        self.pause_on_eos:bool = conf.get("pause_on_eos", False)

        self.writer: cv2.VideoWriter = None
        self.__fps_measured: float = -1

    @property
    def callback(self) -> ImageProcessorCallback:
        return self.__cb

    @callback.setter
    def callback(self, cb: Optional[ImageProcessorCallback]=None) -> None:
        self.__cb = cb if cb is not None else ImageProcessor.__DEFAULT_CALLBACK

    def is_drawing(self) -> None:
        return self.window_name is not None or self.output_video is not None

    @property
    def fps_measured(self) -> float:
        return self.__fps_measured

    def run(self) -> Tuple[int,int,float]:
        last_frame_index = 0
        capture_count = 0
        elapsed_avg = None
        started = time.time()
        self.__fps_measured = 0

        with closing(self.__camera.open()) as cap:
            try:
                self.__setup_window(cap)
                self.writer = self.__setup_video_writer(cap)
                progress = self.__setup_progress(cap)
                self.__cb.on_started(self)
            except Exception as e:
                if self.writer:
                    self.writer.release()
                    self.writer = None
                raise e

            window_name = self.window_name
            key = ''
            while cap.is_open():
                frame: Frame = cap()
                if frame is None:
                    break
                capture_count += 1

                frame = self.__cb.process_image(frame)
                if frame is None:
                    break
                
                if window_name or self.writer:
                    convas = cv2.putText(frame.image, f'frames={frame.index}, fps={self.__fps_measured:.2f}',
                                        (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
                if self.writer:
                    self.writer.write(convas)
                    
                if window_name:
                    cv2.imshow(window_name, convas)
                    key = cv2.waitKey(int(1)) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        while True:
                            key = cv2.waitKey(1000 * 60 * 60) & 0xFF
                            if key == ord(' '):
                                break
                    else:
                        key = self.__cb.set_control(key)

                if progress is not None:
                    progress.update(frame.index - last_frame_index)
                    progress.refresh()
                    last_frame_index = frame.index
                
                elapsed = time.time() - started
                fps = 1 / (elapsed / capture_count)
                weight = ImageProcessor.__ALPHA if capture_count > 10 else 0.5
                self.__fps_measured = weight*fps + (1-weight)*self.__fps_measured

        if key != ord('q') and window_name and self.pause_on_eos:
            cv2.waitKey(-1)

        self.__teardown()
        if progress:
            progress.close()

        return time.time()-started, capture_count, self.__fps_measured

    def __setup_video_writer(self, cap: ImageCapture) -> Optional[cv2.VideoWriter]:
        if self.output_video:
            path = Path(self.output_video)
            fourcc = None
            ext = path.suffix.lower()
            if ext == '.mp4':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif ext == '.avi':
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            else:
                raise IOError("unknown output video file extension: 'f{ext}'")

            path = path.resolve()
            path.parent.mkdir(exist_ok=True)
            return cv2.VideoWriter(str(path), fourcc, cap.fps, cap.size.to_tuple())
        else:
            return None

    def __setup_window(self, cap: ImageCapture) -> None:
        if self.window_name:
            cv2.namedWindow(self.window_name)
            win_size = cap.size
            cv2.resizeWindow(winname=self.window_name, width=win_size.width, height=win_size.height)

    def __setup_progress(self, cap: ImageCapture) -> Optional[tqdm]:
        if self.show_progress:
            total_count = cap.total_frame_count
            return tqdm(total=total_count)
        else:
            return None

    def __teardown(self) -> None:
        try:
            self.__cb.on_stopped()
        except Exception as e:
            # import sys
            # print("********************", e, file=sys.stderr)
            raise e

        if self.writer:
            self.writer.release()
            self.writer = None

        if self.window_name:
            cv2.destroyWindow(self.window_name)