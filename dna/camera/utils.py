
from typing import Optional

from omegaconf import OmegaConf

import dna
from .camera import Size2d, Camera
from .image_processor import ImageProcessor, ImageProcessorCallback


def create_camera_from_conf(conf: OmegaConf):
    from .opencv_camera import OpenCvCamera, OpenCvVideFile

    camera = OpenCvVideFile.from_conf(conf) if OpenCvCamera.is_video_file(conf.uri) \
                                            else OpenCvCamera.from_conf(conf)
    if dna.conf.get_config(conf, 'threaded', False):
        from .threaded_camera import ThreadedCamera
        camera = ThreadedCamera(camera)

    return camera

def create_camera(uri:str, size: Optional[Size2d]=None) -> Camera:
    from .opencv_camera import OpenCvCamera, OpenCvVideFile

    if OpenCvCamera.is_video_file(uri):
        return OpenCvVideFile(uri, size)
    else:
        return OpenCvCamera(uri, size)

def create_image_processor(camera: Camera, conf: OmegaConf) -> ImageProcessor:
    if conf.get('window_name', None) is None:
        conf.window_name = f'camera={camera.uri}' if conf.get('show', False) else None

    return ImageProcessor(camera=camera, conf=conf)