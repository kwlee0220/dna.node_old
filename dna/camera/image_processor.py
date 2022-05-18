from __future__ import annotations

from dataclasses import dataclass, field
from tkinter import W
from typing import Optional, Tuple
from pathlib import Path
from abc import ABCMeta, abstractmethod
import time

from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from dna import color, Frame
from .camera import Camera, ImageCapture


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


class ImageProcessor(metaclass=ABCMeta):
    __ALPHA = 0.2
    __DEFAULT_CALLBACK = DefaultCallback()
    __slots__ = '__cap', '__cb', '__params', 'writer', '__fps_measured'

    @dataclass(frozen=True, eq=True, slots=True)
    class Parameters:
        window_name: Optional[str] = field(default=None)
        output_video: Optional[str] = field(default=None)
        show_progress: bool = field(default=False)
        pause_on_eos: bool = field(default=False)

        @staticmethod
        def from_conf(conf: OmegaConf):
            window_name = conf.get("window_name", None)
            output_video = conf.get("output_video", None)
            show_progress = conf.get("show_progress", False)
            pause_on_eos = conf.get("pause_on_eos", False)
            param = ImageProcessor.Parameters(window_name=window_name, output_video=output_video,
                                            show_progress=show_progress, pause_on_eos=pause_on_eos)
            return param

    def __init__(self, params:Parameters,
                capture:ImageCapture,
                callback:Optional[ImageProcessorCallback]=None):
        self.__cap = capture
        self.__cb = callback if callback else ImageProcessor.__DEFAULT_CALLBACK
        self.__params = params

        self.writer: cv2.VideoWriter = None
        self.__fps_measured: float = -1

    @property
    def image_capture(self) -> ImageCapture:
        return self.__cap

    @property
    def parameters(self) -> Parameters:
        return self.__params

    @property
    def fps_measured(self) -> float:
        return self.__fps_measured

    def run(self) -> Tuple[int,float]:
        last_frame_index = 0
        capture_count = 0
        elapsed_avg = None
        started = time.time()
        self.__fps_measured = 0

        try:
            self.__setup_window()
            self.writer = self.__setup_video_writer()
            progress = self.__setup_progress()
            self.__cb.on_started(self)
        except Exception as e:
            if self.writer:
                self.writer.release()
                self.writer = None
            raise e

        window_name = self.parameters.window_name
        key = ''
        while self.__cap.is_open():
            frame: Frame = self.__cap()
            if frame is None:
                break
            capture_count += 1

            frame = self.__cb.process_image(frame)
            if frame is None:
                break
            
            if window_name:
                convas = cv2.putText(frame.image, f'frames={frame.index}, fps={self.__fps_measured:.2f}',
                                    (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
            if self.writer:
                self.writer.write(convas)
                
            if window_name:
                cv2.imshow(self.parameters.window_name, convas)
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
                progress.update(frame.frame_index - last_frame_index)
                progress.refresh()
                last_frame_index = frame.frame_index
            
            elapsed = time.time() - started
            fps = 1 / (elapsed / capture_count)
            weight = ImageProcessor.__ALPHA if capture_count > 10 else 0.5
            self.__fps_measured = weight*fps + (1-weight)*self.__fps_measured

        if key != ord('q') and window_name and self.parameters.pause_on_eos:
            cv2.waitKey(-1)

        self.__teardown()
        if progress:
            progress.close()

        return time.time()-started, capture_count, self.__fps_measured

    def __setup_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.parameters.output_video:
            path = Path(self.parameters.output_video)
            fourcc = None
            ext = path.suffix.lower()
            if ext == '.mp4':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif ext == '.avi':
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            else:
                raise IOError("unknown output video file extension: 'f{ext}'")

            return cv2.VideoWriter(str(path.resolve()), fourcc, self.__cap.fps, self.__cap.size.to_tuple())
        else:
            return None

    def __setup_window(self) -> None:
        if self.parameters.window_name:
            cv2.namedWindow(self.parameters.window_name)
            win_size = self.__cap.size
            cv2.resizeWindow(winname=self.parameters.window_name, width=win_size.width, height=win_size.height)

    def __setup_progress(self) -> Optional[tqdm]:
        if self.parameters.show_progress:
            total_count = self.__cap.total_frame_count
            return tqdm(total=total_count)
        else:
            return None

    def __teardown(self) -> None:
        try:
            try:
                self.__cb.on_stopped()
            except Exception as e:
                import sys
                print("********************", e, file=sys.stderr)

            if self.writer:
                self.writer.release()
                self.writer = None

            if self.parameters.window_name:
                cv2.destroyWindow(self.parameters.window_name)
        finally:
            self.__cap.close()