from __future__ import annotations
from typing import Optional, Tuple, List
from abc import ABCMeta, abstractmethod
from contextlib import suppress

import logging
from pathlib import Path
import time
from datetime import timedelta

import numpy as np

from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from dna import color, Frame
from dna.types import Size2d
from .camera import ImageCapture
from dna.execution import AbstractExecution, ExecutionContext, NoOpExecutionContext, CancellationError

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
    __slots__ = ('capture', 'conf', 'frame_processors', 'suffix_processors', 'show_processor',
                 '_is_drawing', 'fps_measured', 'logger', 'cond', 'ready')

    from dataclasses import dataclass
    @dataclass(frozen=True)    # slots=True
    class Result:
        elapsed: float
        frame_count: int
        fps_measured: float
        failure_cause: Exception

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

        self.show_processor:ShowFrame = None
        output_video:str = conf.get("output_video", None)
        
        show_str:str = conf.get("show", None)
        show:Optional[Size2d] = Size2d.parse_string(show_str) if show_str is not None else None

        self._is_drawing:bool = show or output_video
        if self._is_drawing:
            self.suffix_processors.append(DrawFrameTitle())

        if output_video:
            self.suffix_processors.append(ImageWriteProcessor(Path(output_video)))

        if show:
            window_name = f'camera={conf.camera.uri}'
            self.show_processor = ShowFrame(window_name, tuple(show.wh) if show else None)
            self.suffix_processors.append(self.show_processor)

        if not isinstance(context, NoOpExecutionContext):
            import dna
            interval = int(dna.conf.get_config(context.request, "progress_report.interval_seconds", 60))
            self.suffix_processors.append(ExecutionProgressReporter(context, interval_secs=interval))

        show_progress:bool = conf.get("show_progress", False)
        if show_progress:
            self.suffix_processors.append(ShowProgress(self.capture.total_frame_count))

        self.fps_measured = 0.
        self.logger = logging.getLogger('dna.image_processor')

        self.ready = False
        
    def close(self) -> None:
        self.stop("close requested", nowait=True)

    def is_drawing(self) -> None:
        return self._is_drawing

    def add_frame_processor(self, frame_proc: FrameProcessor) -> None:
        self.frame_processors.append(frame_proc)
        if self.show_processor:
            self.show_processor.add_processor(frame_proc)

    def wait_for_ready(self) -> None:
        with self.cond:
            while not self.ready:
                self.cond.wait()
        
    def run_work(self) -> Result:
        started = time.time()

        processors = self.frame_processors + self.suffix_processors
        for fproc in processors:
            fproc.on_started(self)

        with self.cond:
            self.ready = True
            self.cond.notify_all()

        capture_count = 0
        self.fps_measured = 0.
        failure_cause = None
        try:
            self.logger.info(f'start: ImageProcess[cap={self.capture}]')
            while self.capture.is_open():
                # 사용자에 의해 동작 멈춤이 요청된 경우 CallationError 예외를 발생시킴
                self.check_stopped()
                
                # ImageCapture에서 처리할 이미지를 읽어 옴.
                frame: Frame = self.capture()
                if frame is None: break
                capture_count += 1

                # 등록된 모든 frame-processor를 capture된 image를 이용해 'process_frame' 메소드를 차례대로 호출한다.
                # process_frame() 호출시 첫번째 processor는 capture된 image를 입력받지만,
                # 이후 processor들은 자신 바로 전에 호출된 process_frame()의 반환값을 입력으로 받는다.
                # 만일 어느 한 frame-processor의 process_frame() 호출 결과가 None인 경우는 이후 frame-processor 호출은 중단되고
                # 전체 image-processor의 수행이 중단된다.
                for fproc in processors:
                    frame = fproc.process_frame(frame)
                    if frame is None: break
                if frame is None: break

                elapsed = time.time() - started
                fps = 1 / (elapsed / capture_count)
                weight = ImageProcessor.__ALPHA if capture_count > 50 else 0.7
                self.fps_measured = weight*fps + (1-weight)*self.fps_measured
        except CancellationError as e:
            failure_cause = e
        except Exception as e:
            failure_cause = e
            self.logger.error(e, exc_info=True)
        finally:
            # 등록된 순서의 역순으로 'on_stopped()' 메소드를 호출함
            for fproc in reversed(processors):
                try:
                    fproc.on_stopped()
                except Exception as e:
                    self.logger.error(e, exc_info=True)

        return ImageProcessor.Result(time.time()-started, capture_count, self.fps_measured, failure_cause)

    def finalize(self) -> None:
        self.capture.close()


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

class DrawFrameTitle(FrameProcessor):
    def on_started(self, proc:ImageProcessor) -> None:
        self.proc = proc
    def on_stopped(self) -> None: pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        convas = cv2.putText(frame.image, f'frames={frame.index}, fps={self.proc.fps_measured:.2f}',
                            (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
        return Frame(image=convas, index=frame.index, ts=frame.ts)

class ShowFrame(FrameProcessor):
    _PAUSE_MILLIS = int(timedelta(hours=1).total_seconds() * 1000)

    def __init__(self, window_name: str, window_size: Optional[Tuple[int,int]]) -> None:
        super().__init__()
        self.window_name = window_name
        self.window_size:Optional[Tuple[int,int]] = window_size if window_size != (0,0) else None
        self.processors: List[FrameProcessor] = []
        self.logger = logging.getLogger('dna.node.frame_processor.show_frame')
        
    def add_processor(self, proc:FrameProcessor) -> None:
        self.processors.append(proc)

    def on_started(self, proc:ImageProcessor) -> None:
        win_size = self.window_size if self.window_size else tuple(proc.capture.size.wh)
        
        self.logger.info(f'create window: {self.window_name}, size=({win_size[0]}x{win_size[1]})')
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, win_size[0], win_size[1])

    def on_stopped(self) -> None:
        self.logger.info(f'destroy window: {self.window_name}')
        with suppress(Exception): cv2.destroyWindow(self.window_name)

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        img = cv2.resize(frame.image, self.window_size, cv2.INTER_AREA) if self.window_size else frame.image
        cv2.imshow(self.window_name, img)

        key = cv2.waitKey(int(1)) & 0xFF
        while True:
            if key == ord('q'):
                self.logger.info(f'interrupted by a key-stroke')
                return None
            elif key == ord(' '):
                self.logger.info(f'paused by a key-stroke')
                while True:
                    key = cv2.waitKey(ShowFrame._PAUSE_MILLIS) & 0xFF
                    if key == ord(' '):
                        self.logger.info(f'resumed by a key-stroke')
                        key = 1
                        break
                    elif key == ord('q'):
                        return None
            elif key != 0xFF:
                for proc in self.processors:
                    key = proc.set_control(key)
                return frame
            else: 
                return frame


class ShowProgress(FrameProcessor):
    __slots__ = ( 'total_frame_count', 'last_frame_index', 'tqdm' )
    
    def __init__(self, total_frame_count: int) -> None:
        super().__init__()
        self.total_frame_count = total_frame_count
        self.last_frame_index = -1
        self.tqdm = None

    def on_started(self, proc:ImageProcessor) -> None:
        pass

    def on_stopped(self) -> None:
        with suppress(Exception):
            self.tqdm.close()
            self.tqdm = None

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        if not self.tqdm:
            self.last_frame_index = 0
            self.tqdm = tqdm(total=self.total_frame_count)
        
        self.tqdm.update(frame.index - self.last_frame_index)
        self.tqdm.refresh()
        self.last_frame_index = frame.index
        return frame
    
    
class ImageWriteProcessor(FrameProcessor):
    def __init__(self, path: Path) -> None:
        super().__init__()
        
        self.path = path.resolve()
        self.logger = logging.getLogger('dna.node.frame_processor.video_writer')

    def on_started(self, proc:ImageProcessor) -> None:
        self.logger.info(f'opening video file: {self.path}')
        
        from .video_writer import VideoWriter
        self.writer = VideoWriter(self.path.resolve(), proc.capture.fps, proc.capture.size)
        self.writer.open()

    def on_stopped(self) -> None:
        self.logger.info(f'closing video file: {self.path}')
        with suppress(Exception): self.writer.close()

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        self.writer.write(frame.image)
        return frame