from __future__ import annotations

from typing import Optional, NewType
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod

from omegaconf import OmegaConf
import numpy as np

import dna
from dna import Size2d, Frame


class Camera(metaclass=ABCMeta):
    @dataclass(frozen=True, eq=True, slots=True)
    class Parameters:
        uri: str|int = field(default=None)
        size: Optional[Size2d] = field(default=None)
        sync: bool = field(default=False)
        begin_frame: int = field(default=1)
        end_frame: Optional[int] = field(default=None)
        threaded: bool = field(default=False)

        @classmethod
        def from_conf(cls, conf:OmegaConf):
            uri = conf.uri
            size = conf.get('size', None)
            size = Size2d.from_conf(size) if size is not None else size
            sync = conf.get("sync", False)
            begin_frame = conf.get("begin_frame", 1)
            end_frame = conf.get("end_frame", None)
            threaded = conf.get("threaded", False)
            
            return cls(uri=uri, size=size, sync=sync, begin_frame=begin_frame, end_frame=end_frame,
                        threaded=threaded)      

    @abstractmethod
    def open(self) -> ImageCapture:
        """Opens a camera
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> Parameters:
        """Returns the parameters for this Camera.

        Returns:
            Parameters: camera parameter
        """
        pass


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
    def repr_str(self) -> str:
        pass