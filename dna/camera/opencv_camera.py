from __future__ import annotations

from typing import Optional, Tuple
import dataclasses
import time
import uuid
from subprocess import CompletedProcess, Popen, STDOUT, DEVNULL

import numpy as np
import cv2
from omegaconf import OmegaConf

import dna
from dna import Size2d, Image, Frame
from .camera import Camera, ImageCapture

class OpenCvCamera(Camera):
    def __init__(self, uri:str, target_size:Optional[Size2d]=None):
        Camera.__init__(self)

        self.__uri = uri
        self.__target_size = target_size
        self.__size = target_size

    @classmethod
    def from_conf(cls, conf:OmegaConf):
        uri = conf.uri
        size_conf = conf.get('size', None)
        size = Size2d.from_conf(size_conf) if size_conf is not None else None

        return cls(uri, size)

    @staticmethod
    def is_local_camera(uri: str):
        return uri.isnumeric()

    @staticmethod
    def is_rtsp_camera(uri: str):
        return uri.startswith('rtsp://')

    @staticmethod
    def is_video_file(uri: str):
        return uri.endswith('.mp4') or uri.endswith('.avi')

    def open(self) -> ImageCapture:
        vid, proc = OpenCvCamera.__open_video_capture(self.uri)
        from_video = self.is_video_file(self.uri)

        if self.__target_size is not None:
            vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.__target_size.width)
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__target_size.height)

        return VideoFileCapture(self, vid) if from_video else OpenCvImageCapture(self, vid, proc)

    @property
    def uri(self) -> str:
        return self.__uri

    def size(self) -> Size2d:
        if self.__target_size is not None:
            return self.__target_size
        
        if self.__size is None:
            vid = self.__open_video_capture(self.uri)
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.__size = Size2d(width, height)
            
        return self.__size

    @staticmethod
    def __open_video_capture(uri: str) -> Tuple[cv2.VideoCapture, CompletedProcess]:
        if uri.isnumeric():
            return (cv2.VideoCapture(int(uri)), None)
        elif uri.startswith('rtsp://'):
            # 만일 OpenCv에서 지원하지 않는 RTSP Url을 사용한다면 replay하는 방법을 해결함.
            if uri.find("&end=") >= 0 or uri.find("start=") >= 0:
                new_uri = f"rtsp://localhost:8554/{uuid.uuid1()}"
                cmd = ["C:\\local\\ffmpeg\\bin\\ffmpeg", "-re", "-rtsp_transport", "tcp", "-i", uri,
                        "-rtsp_transport", "tcp", "-c:v", "copy", "-f", "rtsp", new_uri]
                proc = Popen(cmd, stdout=DEVNULL, stderr=DEVNULL)
                cv2.waitKey(4000)

                vcap = cv2.VideoCapture(new_uri)
                while (1):
                    ret, frame = vcap.read()
                    if ret: return (vcap, proc)
                    else:
                        cv2.waitKey(1000)
                        vcap = cv2.VideoCapture(new_uri)
            else:
                return (cv2.VideoCapture(uri), None)
        elif uri.endswith('.mp4') or uri.endswith('.avi'):  # from video-file
            import os
            if not os.path.exists(uri):
                raise IOError(f"invalid video file path: {uri}")
            return (cv2.VideoCapture(uri), None)
        else:
            raise ValueError(f"invalid camera uri: {uri}")

    def __repr__(self) -> str:
        size_str = f', size={self.__size}' if self.__size is not None else ''
        return f"{__class__.__name__}(uri={self.uri}{size_str})"


class OpenCvVideFile(OpenCvCamera):
    def __init__(self, uri:str, size:Optional[Size2d]=None, sync:bool=True,
                begin_frame:int=1, end_frame:Optional[int]=None):
        OpenCvCamera.__init__(self, uri, size)

        self.sync = sync
        self.begin_frame = begin_frame
        self.end_frame = end_frame

    @classmethod
    def from_conf(cls, conf:OmegaConf):
        uri = conf.uri
        size_conf = conf.get('size', None)
        size = Size2d.from_conf(size_conf) if size_conf is not None else None

        sync = conf.get('sync', True)
        begin_frame = conf.get('begin_frame', 1)
        end_frame = conf.get('end_frame', None)

        return cls(uri, size, sync, begin_frame, end_frame)

    def __repr__(self) -> str:
        size_str = f', size={self.__size}' if self.__size is not None else ''
        return f"{__class__.__name__}(uri={self.uri}{size_str}, sync={self.sync})"

class OpenCvImageCapture(ImageCapture):
    __slots__ = '__camera', '__vid', '__proc', '__size', '__fps', '__frame_index'

    def __init__(self, camera: OpenCvCamera, vid: cv2.VideoCapture, proc: CompletedProcess) -> None:
        """Create a OpenCvImageCapture object.

        Args:
            uri (str): Resource identifier.
        """
        self.__camera = camera
        if vid is None:
            raise ValueError(f'cv2.VideoCapture is invalid')
        self.__vid:cv2.VideoCapture = vid            # None if closed
        self.__proc:CompletedProcess = proc

        width = int(self.__vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.__vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__size = Size2d(width, height)
        self.__fps = self.__vid.get(cv2.CAP_PROP_FPS)
        self.__frame_index = 0

    def close(self) -> None:
        if self.__vid:
            self.__vid.release()
            self.__vid = None
            if self.__proc is not None:
                self.__proc.kill()

    def __call__(self) -> Optional[Frame]:
        if not self.is_open():
            raise IOError(f"{self.__class__.__name__}: not opened")

        ret, mat = self.__vid.read()
        if not ret:
            return None

        self.__frame_index += 1
        return Frame(image=Image(mat), index=self.frame_index, ts=time.time())

    def is_open(self) -> bool:
        return self.__vid is not None

    @property
    def camera(self) -> OpenCvCamera:
        return self.__camera

    @property
    def cv2_video_capture(self) -> cv2.VideoCapture:
        return self.__vid

    @property
    def size(self) -> Size2d:
        return self.__size

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def frame_index(self) -> int:
        return self.__frame_index

    def set_frame_index(self, index:int) -> None:
        self.__vid.set(cv2.CAP_PROP_POS_FRAMES, index-1)
        self.__frame_index = index

    @property
    def repr_str(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return f'{state}, size={self.size}, frames={self.frame_index}, fps={self.fps:.0f}/s'

    def __repr__(self) -> str:
        return f'OpenCvCamera({self.repr_str})'


class VideoFileCapture(OpenCvImageCapture):
    __ALPHA = 0.5
    __OVERHEAD = 0.02
    __slots__ = '__interval', '__sync', '__last_img', '__overhead'

    def __init__(self, camera: OpenCvVideFile, vid: cv2.VideoCapture) -> None:
        """Create a VideoFileCapture object.

        Args:
            camera (OpenCvCamera): Resource identifier.
        """
        super().__init__(camera, vid, None)

        self.__interval = 1.0 / self.fps
        self.__sync = self.camera.sync
        self.__last_frame = Frame(image=None, index=-1, ts=time.time())
        self.__overhead = 0.0

        if self.camera.begin_frame > 1:
            self.set_frame_index(self.camera.begin_frame)
            
        import sys
        self.__last_frame_index = sys.maxsize if self.camera.end_frame is None else self.camera.end_frame

    @property
    def total_frame_count(self) -> int:
        return int(self.cv2_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __call__(self) -> Optional[Frame]:
        started = self.__last_frame.ts

        frame: Frame = super().__call__()
        if frame is None:
            return frame
        
        # 지정된 마지막 프레임 번호보다 큰 경우는 image capture를 종료시킨다.
        if frame.index > self.__last_frame_index:
            return None

        if self.__sync:
            remains = self.__interval - (frame.ts - started) - self.__overhead
            # print(f'elapsed={(img.ts - started)*1000:.0f}, overhead={self.__overhead*1000:.0f}, remains={remains*1000:.0f}')
            if remains > 0.005:
                time.sleep(remains)

                # System상 정확히 remains 초 만큼 대기하지 않기 때문에 그 차이를 계산해서
                # 다음번 대기 시간에 반영시킴.
                ts = time.time()
                overhead = (ts - started) - self.__interval
                if self.__overhead == 0: # for the initial frame
                    self.__overhead = overhead
                else:
                    self.__overhead == (VideoFileCapture.__ALPHA * overhead) + ((1-VideoFileCapture.__ALPHA)*self.__overhead)
                
                frame = dataclasses.replace(frame, ts=ts)

        self.__last_frame = frame
        return frame

    @property
    def repr_str(self) -> str:
        return f'{super().repr_str}, sync={self.__sync}'