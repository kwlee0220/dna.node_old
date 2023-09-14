from __future__ import annotations

from typing import Union, Optional
import dataclasses
import time
import uuid
from subprocess import CompletedProcess, Popen, DEVNULL

import numpy as np
import cv2
from omegaconf import OmegaConf

import dna
from dna import Size2d, Image, Frame
from dna.utils import utc_now_seconds
from .camera import Camera, ImageCapture


def create_opencv_camera(uri:str, **options) -> OpenCvCamera:
    """Create an OpenCvCamera object of the given URI.
    The additional options will be given by dictionary ``options``.
    The options contain the followings:
    - size: the size of the image that the created camera will capture (optional)

    Args:
        uri (str): id of the camera.

    Returns:
        OpenCvCamera: an OpenCvCamera object.
        If URI points to a video file, ``OpenCvVideFile`` object is returned. Otherwise,
        ``OpenCvCamera`` is returned.
    """
    if OpenCvCamera.is_video_file(uri):
        if 'open_ts' in options:
            return TestingVideoFile(uri, **options)
        else:
            return OpenCvVideFile(uri, **options)
    elif OpenCvCamera.is_local_camera(uri):
        return OpenCvCamera(uri, **options)
    elif OpenCvCamera.is_rtsp_camera(uri):
        return RTSPOpenCvCamera(uri, **options)
    else:
        raise ValueError(f'invalid OpenCvCamera URI: {uri}')


def create_opencv_camera_from_conf(conf:OmegaConf) -> OpenCvCamera:
    """Create a camera from OmegaConf configuration.

    Args:
        conf (OmegaConf): OmegaConf configuration.

    Returns:
        OpenCvCamera: an OpenCvCamera object.
    """
    options = {k:v for k, v in dict(conf).items() if k != 'uri'}
    return create_opencv_camera(conf.uri, **options)


class OpenCvCamera(Camera):
    def __init__(self, uri:str, **options:object):
        Camera.__init__(self)

        self._uri = uri
        self._size = Size2d.from_expr(options.get('size'))
        self._target_size = self._size
        self.open_ts = options.get('open_ts', 0)

    @staticmethod
    def is_local_camera(uri:str):
        '''Determines that the give URI is for the local camera or not.
        'Local camera' means the one is directly connected to this computer through USB or any other media.'''
        return uri.isnumeric()

    @staticmethod
    def is_rtsp_camera(uri:str):
        '''Determines whether the camera of the give URI is a remote one accessed by the RTSP protocol.'''
        return uri.startswith('rtsp://')

    @staticmethod
    def is_video_file(uri:str):
        '''Determines whether the images captured from a video file or not.'''
        return uri.endswith('.mp4') or uri.endswith('.avi')

    def open(self) -> ImageCapture:
        """Open this camera and set ready for captures.

        Returns:
            ImageCapture: a image capturing session from this camera.
        """
        uri = int(self.uri) if OpenCvCamera.is_local_camera(self.uri) else self.uri
        vid = self._open_video_capture(uri)

        from_video = self.is_video_file(self.uri)
        # return VideoFileCapture(self, vid) if from_video else OpenCvImageCapture(self, vid, proc)
        return TestingVideoFileCapture(self, vid, self.open_ts) if from_video else OpenCvImageCapture(self, vid)

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def size(self) -> Size2d:
        if self._target_size is not None:
            return self._target_size
        
        if self._size is None:
            # if the image size has not been set yet, open this camera and find out the image size.
            vid = self._open_video_capture(self._uri)
            try:
                width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._size = Size2d([width, height])
            finally:
                vid.release()
            
        return self._size
    
    def _open_video_capture(self, uri:Union[str,int]) -> cv2.VideoCapture:
        vid = cv2.VideoCapture(uri)
        if self._target_size is not None:
            vid.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_size.width)
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_size.height)
            
        return vid

    def __repr__(self) -> str:
        size_str = f', size={self._size}' if self._size is not None else ''
        return f"{__class__.__name__}(uri={self.uri}{size_str})"


class OpenCvVideFile(OpenCvCamera):
    def __init__(self, uri:str, **options):
        super().__init__(uri, **options)

        self.sync = options.get('sync', True)
        self.begin_frame = options.get('begin_frame', 1)
        self.end_frame = options.get('end_frame')

    def open(self) -> VideoFileCapture:
        import os
        if not os.path.exists(self.uri):
            raise IOError(f"invalid video file path: {self.uri}")
        
        vid = self._open_video_capture(self.uri)
        return VideoFileCapture(self, vid)
        # return TestingVideoFileCapture(self, vid, self.open_ts)

    def __repr__(self) -> str:
        size_str = f', size={self._size}' if self._size is not None else ''
        return f"{__class__.__name__}(uri={self.uri}{size_str}, sync={self.sync})"


class OpenCvImageCapture(ImageCapture):
    __slots__ = '_camera', '_vid', '_size', '_fps', '_frame_index'

    def __init__(self, camera:OpenCvCamera, vid:cv2.VideoCapture) -> None:
        """Create a OpenCvImageCapture object.

        Args:
            uri (str): Resource identifier.
        """
        self._camera = camera
        if vid is None:
            raise ValueError(f'cv2.VideoCapture is invalid')
        self._vid:cv2.VideoCapture = vid            # None if closed

        width = int(self._vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._size = Size2d([width, height])
        self._fps = self._vid.get(cv2.CAP_PROP_FPS)
        self._frame_index = 0

    def close(self) -> None:
        if self._vid:
            self._vid.release()
            self._vid = None

    def __call__(self) -> Optional[Frame]:
        if not self.is_open():
            raise IOError(f"{self.__class__.__name__}: not opened")

        ret, mat = self._vid.read()
        if not ret:
            return None

        self._frame_index += 1
        return Frame(image=Image(mat), index=self.frame_index, ts=utc_now_seconds())

    def is_open(self) -> bool:
        return self._vid is not None

    @property
    def camera(self) -> OpenCvCamera:
        return self._camera

    @property
    def cv2_video_capture(self) -> cv2.VideoCapture:
        return self._vid

    @property
    def size(self) -> Size2d:
        return self._size

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def sync(self) -> bool:
        return False

    @property
    def repr_str(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return f'{state}, size={self.size}, fps={self.fps:.0f}/s'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.repr_str})'
    

class VideoFileCapture(OpenCvImageCapture):
    __ALPHA = 0.3
    __slots__ = '_interval', 'sync', '_last_captured_ts', '_end_frame_index', '_overhead'

    def __init__(self, camera:OpenCvVideFile, vid:cv2.VideoCapture) -> None:
        super().__init__(camera, vid)

        self._interval = 1.0 / self.fps
        self.sync = self.camera.sync
        self._last_captured_ts = utc_now_seconds()
        self._overhead = 0.0

        if self._camera.begin_frame > 1:
            self.frame_index = self._camera.begin_frame - 1
            
        if self._camera.end_frame:
            self._end_frame_index = self._camera.end_frame
        else:
            self._end_frame_index = int(self.cv2_video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) + 100

    @property
    def total_frame_count(self) -> int:
        return int(self.cv2_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @frame_index.setter
    def frame_index(self, index:int) -> None:
        if index <= 0 and index > self.total_frame_count:
            raise ValueError(f'index({index}) should be between 1 and {self.total_frame_count}')
        
        self._vid.set(cv2.CAP_PROP_POS_FRAMES, index)
        self._frame_index = index

    def __call__(self) -> Optional[Frame]:
        started_ts = self._last_captured_ts

        frame: Frame = super().__call__()
        if frame is None:
            return frame
        
        # 지정된 마지막 프레임 번호보다 큰 경우는 image capture를 종료시킨다.
        if frame.index >= self._end_frame_index:
            return None

        if self.sync:
            remains = self._interval - (frame.ts - started_ts) - self._overhead - 0.008
            if remains > 0:
                time.sleep(remains)

                # System상 정확히 remains 초 만큼 대기하지 않기 때문에 그 차이를 계산해서
                # 다음번 대기 시간에 반영시킴.
                now = utc_now_seconds()
                overhead = (now - started_ts) - self._interval
                if self._overhead == 0: # for the initial frame
                    self._overhead = overhead
                else:
                    self._overhead = (VideoFileCapture.__ALPHA * overhead) + ((1-VideoFileCapture.__ALPHA)*self._overhead)
                
                frame = dataclasses.replace(frame, ts=now)

        self._last_captured_ts = frame.ts
        return frame

    @property
    def repr_str(self) -> str:
        return f'{super().repr_str}, sync={self.sync}'
    
    
class RTSPOpenCvCamera(OpenCvCamera):
    def __init__(self, uri:str, **options):
        super().__init__(uri, **options)
        
        self.ffmpeg_cmd = None
        if uri.find("&end=") >= 0 or uri.find("start=") >= 0:
            ffmpeg_path = options.get('ffmpeg_path')
            if not ffmpeg_path:
                raise ValueError(f'FFMPEG command is not specified')
            
            self.ffmpeg_cmd = [ffmpeg_path, "-re", "-rtsp_transport", "tcp", "-i", uri,
                               "-rtsp_transport", "tcp", "-c:v", "copy", "-f", "rtsp"]
            
    def open(self) -> OpenCvImageCapture:
        if self.ffmpeg_cmd:
            ffmpeg_cmd = self.ffmpeg_cmd.copy()
            
            new_uri = f"rtsp://localhost:8554/{uuid.uuid1()}"
            # new_uri = f"rtsp://localhost:8554/visual"
            ffmpeg_cmd.append(new_uri)
            proc = Popen(ffmpeg_cmd, stdout=DEVNULL, stderr=DEVNULL)
            cv2.waitKey(5000)

            while True:
                vcap = self._open_video_capture(new_uri)
                ret, _ = vcap.read()
                if ret:
                    return RTSPOpenCvImageCapture(self, vcap, proc)
                else:
                    cv2.waitKey(1000)
        else:
            return super().open()

    def __repr__(self) -> str:
        size_str = f', size={self._size}' if self._size is not None else ''
        ffmpeg_str = f', ffmpeg={self.ffmpeg_path}' if self.ffmpeg_path else ""
        return f"{__class__.__name__}(uri={self.uri}{size_str}, sync={self.sync}{ffmpeg_str})"

class RTSPOpenCvImageCapture(OpenCvImageCapture):
    __slots__ = ( '_proc', )

    def __init__(self, camera:RTSPOpenCvCamera, vid:cv2.VideoCapture, proc:CompletedProcess) -> None:
        super().__init__(camera, vid)
        
        self._proc:CompletedProcess = proc
        
    def close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
        super().close()

    def __repr__(self) -> str:
        return f'RTSPCapture({self.repr_str})'

    
class TestingVideoFile(OpenCvVideFile):
    def __init__(self, uri:str, **options):
        super().__init__(uri, **options)
        
        self.open_ts = float(options.get('open_ts', 0))

    def open(self) -> TestingVideoFileCapture:
        vfc = super().open()
        return TestingVideoFileCapture(camera=self, vid=vfc._vid, open_ts=self.open_ts)
        

class TestingVideoFileCapture(VideoFileCapture):
    def __init__(self, camera: OpenCvVideFile, vid: cv2.VideoCapture, open_ts:float) -> None:
        super().__init__(camera, vid)
        
        self.last_ts:float = open_ts
        self.inc_secs:float =  1 / self.fps

    def __call__(self) -> Optional[Frame]:
        frame: Frame = super().__call__()
        if frame is None:
            return frame
        
        # self.last_ts += self.inc_secs
        ts = frame.index / 10
        frame = dataclasses.replace(frame, ts=ts)
        return frame