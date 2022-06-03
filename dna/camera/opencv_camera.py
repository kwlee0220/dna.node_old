from __future__ import annotations

from typing import Optional, Tuple
import dataclasses
import time

import numpy as np
import cv2
from omegaconf import OmegaConf

import dna
from dna import Size2d, Image, Frame
from .camera import Camera, ImageCapture



class OpenCvCamera(Camera):
    # def __init__(self, conf:OmegaConf):
    #     Camera.__init__(self)

    #     self.__uri = conf.uri

    #     size_conf = conf.get('size', None)
    #     self.__size = Size2d.from_conf(size_conf) if size_conf is not None else None
    def __init__(self, uri:str, size:Optional[Size2d]=None):
        Camera.__init__(self)

        self.__uri = uri
        self.__size = size

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

    def __open_cv_capture(self) -> Tuple[cv2.VideoCapture,bool]:
        uri:str = self.uri
        live:bool = True

        if uri.isnumeric():
            vid = cv2.VideoCapture(int(uri))
        elif uri.startswith('rtsp://'):
            vid = cv2.VideoCapture(uri)
        elif uri.endswith('.mp4') or uri.endswith('.avi'):  # from video-file
            import os
            if not os.path.exists(uri):
                raise IOError(f"invalid video file path: {uri}")
            vid = cv2.VideoCapture(uri)
            live = False
        else:
            raise ValueError(f"invalid camera uri: {uri}")

        if self.size is not None:
            vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.size.width)
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size.height)

        return vid, live

    def open(self) -> ImageCapture:
        from_video = False
        if self.uri.isnumeric():
            vid = cv2.VideoCapture(int(self.uri))
        elif self.uri.startswith('rtsp://'):
            vid = cv2.VideoCapture(self.uri)
        elif self.uri.endswith('.mp4') or self.uri.endswith('.avi'):  # from video-file
            import os
            if not os.path.exists(self.uri):
                raise IOError(f"invalid video file path: {self.uri}")
            vid = cv2.VideoCapture(self.uri)
            from_video = True
        else:
            raise ValueError(f"invalid camera uri: {self.uri}")

        if self.size is not None:
            vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.size.width)
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size.height)

        return VideoFileCapture(self, vid) if from_video else OpenCvImageCapture(self, vid)

    @property
    def uri(self) -> str:
        return self.__uri

    @property
    def size(self) -> Optional[Size2d]:
        return self.__size

    def __repr__(self) -> str:
        size_str = f', size={self.size}' if self.__size is not None else ''
        return f"{__class__.__name__}(uri={self.uri}{size_str})"


class OpenCvVideFile(OpenCvCamera):
    # def __init__(self, conf:OmegaConf):
    #     OpenCvCamera.__init__(self, conf)

    #     self.sync = conf.get('sync', True)
    #     self.begin_frame = conf.get('begin_frame', 1)
    #     self.end_frame = conf.get('end_frame', None)

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
        size_str = f', size={self.size}' if self.size is not None else ''
        return f"{__class__.__name__}(uri={self.uri}{size_str}, sync={self.sync})"


class OpenCvImageCapture(ImageCapture):
    __slots__ = '__camera', '__vid', '__size', '__fps', '__frame_index'

    def __init__(self, camera: OpenCvCamera, vid: cv2.VideoCapture) -> None:
        """Create a OpenCvImageCapture object.

        Args:
            uri (str): Resource identifier.
        """
        self.__camera = camera
        if vid is None:
            raise ValueError(f'cv2.VideoCapture is invalid')
        self.__vid = vid            # None if closed

        width = int(self.__vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.__vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__size = Size2d(width, height)
        self.__fps = self.__vid.get(cv2.CAP_PROP_FPS)
        self.__frame_index = 0

    def close(self) -> None:
        if self.__vid:
            self.__vid.release()
            self.__vid = None

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
        super().__init__(camera, vid)

        self.__interval = 1.0 / self.fps
        self.__sync = self.camera.sync
        self.__last_frame = Frame(image=None, index=-1, ts=time.time())
        self.__overhead = 0.0

        if self.camera.begin_frame > 1:
            self.set_frame_index(self.camera.begin_frame)

    @property
    def total_frame_count(self) -> int:
        return int(self.cv2_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __call__(self) -> Optional[Frame]:
        started = self.__last_frame.ts

        frame: Frame = super().__call__()
        if frame is None:
            return frame

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