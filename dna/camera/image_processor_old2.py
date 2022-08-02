from __future__ import annotations
from typing import Optional, Tuple
from abc import ABCMeta, abstractmethod

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
LOGGER = logging.getLogger(__name__)


class ImageProcessContext(ExecutionContext):
    def __init__(self, parent_context: ExecutionContext=NoOpExecutionContext()) -> None:
        super().__init__()

        self.parent = parent_context
        self.started_time:time = None
        self.report_interval:timedelta = None

    def started(self) -> None:
        self.next_report_time = None
        
        # report interval이 설정된 경우, 다음 report 시각을 설정함.
        if self.report_interval is not None:
            self.next_report_time = time.time() + self.report_interval
            
        if self.parent is not None:
            self.parent.started()
        LOGGER.info(f'started')

    def report_progress(self, frame:Frame) -> None:
        if self.parent is not None and self.next_report_time is not None:
            now = time.time()
            if now >= self.next_report_time:
                progress = {
                    'frame_index': frame.index
                }
                self.parent.report_progress(progress)
                LOGGER.info(f'report: progress={progress}')
                self.next_report_time += self.report_interval

    def completed(self, result: Tuple[int, int, float]) -> None:
        if self.parent is not None:
            result = {
                'elapsed_millis': result[0],
                'image_count': result[1],
                'fps_measured': result[2]
            }
            self.parent.completed(result)
            LOGGER.info(f'completed: result={result}')

    def stopped(self, details:str) -> None:
        if self.parent is not None:
            self.parent.stopped(details)
            LOGGER.info(f'stopped: details={details}')

    def failed(self, cause:str) -> None:
        if self.parent is not None:
            self.parent.failed(cause)
            LOGGER.info(f'failed: cause={cause}')


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


class ImageProcessor(AbstractExecution):
    __ALPHA = 0.2
    __DEFAULT_CALLBACK = DefaultCallback()
    __slots__ = ('capture', 'conf', 'callback', 'window_name', 'output_video')

    def __init__(self, cap: ImageCapture, conf: OmegaConf, context: Optional[ExecutionContext]=None):
        super().__init__(context)
        
        self.capture = cap
        self.conf = conf
        self.callback = ImageProcessor.__DEFAULT_CALLBACK

        self.window_name:str = conf.get("window_name", None)
        self.output_video:str = conf.get("output_video", None)

    def is_drawing(self) -> None:
        return self.window_name is not None or self.output_video is not None
        
    def run_work(self) -> Tuple[float,int,float]:
        self.setup_window()
        tqdm = self.setup_tqdm()
        video_writer = self.setup_video_writer()
        pause_on_eos = self.conf.get("pause_on_eos", False)
        
        try:
            self.callback.on_started(self)
        except Exception as e:
            if video_writer:
                video_writer.release()
            raise e
            
        started = time.time()
        last_frame_index = 0
        capture_count = 0
        fps_measured = 0.
        
        LOGGER.info(f'start: ImageProcess[cap={self.capture}]')
        while self.capture.is_open():
            # 사용자에 의해 동작 멈춤이 요청된 경우 CallationError 예외를 발생시킴
            self.check_stopped()
            
            # ImageCapture에서 처리할 이미지를 읽어 옴.
            frame: Frame = self.capture()
            if frame is None:
                break
            capture_count += 1

            frame = self.callback.process_image(frame)
            if frame is None:
                break
            
            if self.window_name or video_writer:
                # 처리된 이미지를 화면에 출력하거나, 출력 video로 출력하는 경우
                # 처리 과정 정보를 이미지를 write함.
                convas = cv2.putText(frame.image, f'frames={frame.index}, fps={fps_measured:.2f}',
                                    (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
                if video_writer:
                    video_writer.write(convas)
                        
                if self.window_name:
                    cv2.imshow(self.window_name, convas)
                    key = cv2.waitKey(int(1)) & 0xFF
                    if key == ord('q'):
                        LOGGER.info(f'interrupted by a key-stroke')
                        break
                    elif key == ord(' '):
                        LOGGER.info(f'paused by a key-stroke')
                        while True:
                            key = cv2.waitKey(1000 * 60 * 60) & 0xFF
                            if key == ord(' '):
                                LOGGER.info(f'resumed by a key-stroke')
                                break
                    else:
                        key = self.callback.set_control(key)
            
            if tqdm is not None:
                tqdm.update(frame.index - last_frame_index)
                tqdm.refresh()
            last_frame_index = frame.index
            self.context().report_progress(frame)
            
            elapsed = time.time() - started
            fps = 1 / (elapsed / capture_count)
            weight = ImageProcessor.__ALPHA if capture_count > 10 else 0.5
            fps_measured = weight*fps + (1-weight)*fps_measured

        if key != ord('q') and self.window_name and pause_on_eos:
            LOGGER.info(f'waiting for the final key-stroke')
            cv2.waitKey(-1)
            
        try:
            self.callback.on_stopped()
            
            if video_writer:
                video_writer.release()
            if tqdm is not None:
                tqdm.close()
            if self.window_name:
                cv2.destroyWindow(self.window_name)
        except Exception as e:
            # import sys
            # print("********************", e, file=sys.stderr)
            raise e

        return time.time()-started, capture_count, fps_measured

    def setup_video_writer(self) -> Optional[cv2.VideoWriter]:
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
            return cv2.VideoWriter(str(path), fourcc, self.capture.fps, self.capture.size.to_tuple())
        else:
            return None

    def setup_window(self) -> None:
        if self.window_name:
            cv2.namedWindow(self.window_name)
            win_size = self.capture.size
            cv2.resizeWindow(winname=self.window_name, width=win_size.width, height=win_size.height)
    
    def setup_tqdm(self) -> Optional[tqdm]:
        if self.conf.get("show_progress", False):
            total_count = self.capture.total_frame_count
            return tqdm(total=total_count)
        else:
            return None