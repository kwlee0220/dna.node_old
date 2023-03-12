from __future__ import annotations

from typing import Optional, Union
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod

from omegaconf import OmegaConf
import numpy as np

import dna
from dna import Size2d, Frame


@dataclass(frozen=True, eq=True)    # slots=True
class Parameters:
    uri: Union[str,int] = field(default=None)
    size: Optional[Size2d] = field(default=None)
    sync: bool = field(default=False)
    begin_frame: int = field(default=1)
    end_frame: Optional[int] = field(default=None)
    threaded: bool = field(default=False)

    @staticmethod
    def from_conf(cls, conf:OmegaConf):
        uri = conf.uri
        size = conf.get('size', None)
        size = Size2d.from_conf(size) if size is not None else size
        sync = conf.get("sync", False)
        begin_frame = conf.get("begin_frame", 1)
        end_frame = conf.get("end_frame", None)
        threaded = conf.get("threaded", False)
        
        return Parameters(uri=uri, size=size, sync=sync, begin_frame=begin_frame, end_frame=end_frame,
                          threaded=threaded)


class Camera(metaclass=ABCMeta):
    @abstractmethod
    def open(self) -> ImageCapture:
        """Opens a camera
        """
        pass

    @property
    @abstractmethod
    def uri(self) -> str:
        """Returns the URI of this Camera.

        Returns:
            Parameters: camera URI
        """
        pass

    @abstractmethod
    def size(self) -> Size2d:
        """Returns the image size captured from this Camera.

        Returns:
            Parameters: the image size captured from this Camera
        """
        pass

    def resize(self, size:Size2d) -> Camera:
        """Returns a Camera that captures the resized images.

        Args:
            size (Size2d): target image size.

        Returns:
            Camera: Camera
        """
        from .resized_camera import ResizingCamera
        return ResizingCamera(self, size)


class ImageCapture(metaclass=ABCMeta):
    @abstractmethod
    def close(self) -> None:
        """Closes this ImageCapture.
        """
        pass

    @abstractmethod
    def is_open(self) -> bool:
        """Returns whether this is opened or not.

        Returns:
            bool: True if this is opened, False otherwise.
        """
        pass

    @abstractmethod
    def __call__(self) -> Optional[Frame]:
        """Captures an OpenCV image frame.

        Returns:
            Frame: captured image frame.
        """
        pass

    @property
    @abstractmethod
    def size(self) -> Size2d:
        """Returns the size of the images that this ImageCapture captures.

        Returns:
            Size2d: (width, height)
        """
        pass

    @property
    @abstractmethod
    def fps(self) -> int:
        """Returns the fps of this ImageCapture.

        Returns:
            int: frames per second.
        """
        pass

    @property
    @abstractmethod
    def frame_index(self) -> int:
        """Returns the total count of images this ImageCapture captures so far.

        Returns:
            int: The number of frames
        """
        pass

    @property
    @abstractmethod
    def sync(self) -> bool:
        """Returns whether frames are captured on its fps or immediately as they are captured from the camera.

        Returns:
            bool: synchronized capture or not
        """
        pass

    @property
    @abstractmethod
    def repr_str(self) -> str:
        pass