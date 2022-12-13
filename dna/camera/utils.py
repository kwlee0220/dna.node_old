
from typing import Optional

from omegaconf import OmegaConf

import dna
from dna import Size2d
from .opencv_camera import OpenCvCamera, OpenCvVideFile
from .threaded_camera import ThreadedCamera

def create_camera(uri:str, size:Size2d=None, begin_frame: int=0) -> OpenCvCamera:
    conf = OmegaConf.create()
    conf.uri = uri
    conf.begin_frame = begin_frame
    return create_camera_from_conf(conf)

def create_camera_from_conf(conf: OmegaConf) -> OpenCvCamera:
    camera = OpenCvVideFile.from_conf(conf) if OpenCvCamera.is_video_file(conf.uri) \
                                            else OpenCvCamera.from_conf(conf)
    if dna.conf.get_config(conf, 'threaded', False):
        camera = ThreadedCamera(camera)

    return camera