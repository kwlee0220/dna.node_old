
from typing import Optional

from omegaconf import OmegaConf

import dna
from .opencv_camera import OpenCvCamera, OpenCvVideFile
from .threaded_camera import ThreadedCamera


def create_camera_from_conf(conf: OmegaConf):
    camera = OpenCvVideFile.from_conf(conf) if OpenCvCamera.is_video_file(conf.uri) \
                                            else OpenCvCamera.from_conf(conf)
    if dna.conf.get_config(conf, 'threaded', False):
        camera = ThreadedCamera(camera)

    return camera