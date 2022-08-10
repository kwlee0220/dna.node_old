from __future__ import annotations
from typing import Optional, Tuple, List
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from contextlib import suppress

import logging
from pathlib import Path
import time
from datetime import timedelta

from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from dna import color, Frame
from .camera import ImageCapture
from dna.execution import AbstractExecution, ExecutionContext, NoOpExecutionContext

import logging

class FrameProcessor(metaclass=ABCMeta):
    @abstractmethod
    def on_started(self, proc:ImageProcessor) -> None:
        pass

    @abstractmethod
    def on_stopped(self) -> None:
        pass

    @abstractmethod
    def process_frame(self, frame:Frame) -> Optional[Frame]:
        pass

    def set_control(self, key:int) -> int:
        return key

class ImageProcessor(AbstractExecution):
    __ALPHA = 0.2
    __slots__ = ('capture', 'conf', 'frame_processors', 'suffix_processors', '_is_drawing', 'fps_measured')

    from dataclasses import dataclass
    @dataclass(frozen=True)    # slots=True
    class Result:
        elapsed: float
        frame_count: int
        fps_measured: float

        def __repr__(self):
            return (f"elapsed={timedelta(seconds=self.elapsed)}, "
                    f"frame_count={self.frame_count}, "
                    f"fps={self.fps_measured:.1f}")

    def __init__(self, cap: ImageCapture, conf: OmegaConf, context: ExecutionContext=NoOpExecutionContext()):
        super().__init__(context=context)
        
        self.capture = cap
        self.conf = conf
        self.frame_processors: List[FrameProcessor] = []
        self.suffix_processors: List[FrameProcessor] = []

        output_video:str = conf.get("output_video", None)
        show:bool = conf.get("show", False)

        self._is_drawing:bool = show or output_video is not None
        if self._is_drawing:
            self.suffix_processors.append(DrawText())

        if output_video is not None:
            self.suffix_processors.append(VideoWriter(output_video))

        if show:
            window_name = f'camera={conf.camera.uri}'
            self.suffix_processors.append(ShowFrame(window_name))

        if not isinstance(context, NoOpExecutionContext):
            import dna
            interval = int(dna.conf.get_config(context.request, "progress_report.interval_seconds", 60))
            self.suffix_processors.append(ExecutionProgressReporter(context, interval_secs=interval))

        show_progress:bool = conf.get("show_progress", False)
        if show_progress:
            self.suffix_processors.append(ShowProgress(self.capture.total_frame_count))

        self.logger = logging.getLogger('dna.image_processor')
        
    def close(self) -> None:
        self.stop("close requested", nowait=True)

    def is_drawing(self) -> None:
        return self._is_drawing

    def add_frame_processor(self, frame_proc: FrameProcessor) -> None:
        self.frame_processors.append(frame_proc)
        
    def run_work(self) -> Result:
        started = time.time()

        processors = self.frame_processors + self.suffix_processors
        for fproc in processors:
            fproc.on_started(self)

        capture_count = 0
        self.fps_measured = 0.
        try:
            self.logger.info(f'start: ImageProcess[cap={self.capture}]')
            while self.capture.is_open():
                # 사용자에 의해 동작 멈춤이 요청된 경우 CallationError 예외를 발생시킴
                self.check_stopped()
                
                # ImageCapture에서 처리할 이미지를 읽어 옴.
                frame: Frame = self.capture()
                if frame is None: break
                capture_count += 1

                for fproc in processors:
                    frame = fproc.process_frame(frame)
                    if frame is None: break
                if frame is None: break

                elapsed = time.time() - started
                fps = 1 / (elapsed / capture_count)
                weight = ImageProcessor.__ALPHA if capture_count > 10 else 0.5
                self.fps_measured = weight*fps + (1-weight)*self.fps_measured
        except Exception as e:
            self.logger.error(e, exc_info=True)
        finally:
            for fproc in reversed(processors):
                try:
                    fproc.on_stopped()
                except Exception as e:
                    self.logger.error(e, exc_info=True)

        return ImageProcessor.Result(time.time()-started, capture_count, self.fps_measured)

class ExecutionProgressReporter(FrameProcessor):
    def __init__(self, context: ExecutionContext, interval_secs: int) -> None:
        super().__init__()
        self.ctx = context
        self.report_interval = interval_secs

    def on_started(self, proc:ImageProcessor) -> None:
        self.next_report_time = time.time() + self.report_interval

    def on_stopped(self) -> None: pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        now = time.time()
        if now >= self.next_report_time:
            progress = {
                'frame_index': frame.index
            }
            self.ctx.report_progress(progress)
            self.next_report_time += self.report_interval
        return frame

class DrawText(FrameProcessor):
    def on_started(self, proc:ImageProcessor) -> None:
        self.proc = proc
    def on_stopped(self) -> None: pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        convas = cv2.putText(frame.image, f'frames={frame.index}, fps={self.proc.fps_measured:.2f}',
                            (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
        return Frame(image=convas, index=frame.index, ts=frame.ts)

class ShowFrame(FrameProcessor):
    _PAUSE_MILLIS = timedelta(hours=1).total_seconds() * 1000

    def __init__(self, window_name: str) -> None:
        super().__init__()
        self.window_name = window_name
        self.logger = logging.getLogger('dna.node.frame_processor.show_frame')

    def on_started(self, proc:ImageProcessor) -> None:
        self.logger.info(f'create window: {self.window_name}')
        cv2.namedWindow(self.window_name)
        win_size = proc.capture.size
        cv2.resizeWindow(winname=self.window_name, width=win_size.width, height=win_size.height)

    def on_stopped(self) -> None:
        self.logger.info(f'destroy window: {self.window_name}')
        with suppress(Exception): cv2.destroyWindow(self.window_name)

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        cv2.imshow(self.window_name, frame.image)
        key = cv2.waitKey(int(1)) & 0xFF
        if key == ord('q'):
            self.logger.info(f'interrupted by a key-stroke')
            return None
        elif key == ord(' '):
            self.logger.info(f'paused by a key-stroke')
            while True:
                key = cv2.waitKey(ShowFrame._PAUSE_MILLIS) & 0xFF
                if key == ord(' '):
                    self.logger.info(f'resumed by a key-stroke')
                    return frame
        else:
            return frame

class ShowProgress(FrameProcessor):
    def __init__(self, total_frame_count: int) -> None:
        super().__init__()
        self.total_frame_count = total_frame_count

    def on_started(self, proc:ImageProcessor) -> None:
        self.last_frame_index = 0
        self.tqdm = tqdm(total=self.total_frame_count)

    def on_stopped(self) -> None:
        with suppress(Exception): self.tqdm.close()

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        self.tqdm.update(frame.index - self.last_frame_index)
        self.tqdm.refresh()
        self.last_frame_index = frame.index
        return frame

class VideoWriter(FrameProcessor):
    def __init__(self, path: Path) -> None:
        super().__init__()

        self.fourcc = None
        ext = path.suffix.lower()
        if ext == '.mp4':
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif ext == '.avi':
            self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        else:
            raise IOError("unknown output video file extension: 'f{ext}'")
        self.path = path.resolve()
        self.logger = logging.getLogger('dna.node.frame_processor.video_writer')

    def on_started(self, proc:ImageProcessor) -> None:
        self.logger.info(f'opening video file: {self.path}')

        self.path.parent.mkdir(exist_ok=True)
        self.video_writer = cv2.VideoWriter(str(self.path), self.fourcc, self.capture.fps, self.capture.size.to_tuple())

    def on_stopped(self) -> None:
        self.logger.info(f'closing video file: {self.path}')
        with suppress(Exception): self.video_writer.release()

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        self.video_writer.write(frame.image)
        return frame