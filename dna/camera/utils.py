
from typing import Optional

import dna
from .camera import Size2d, Camera, ImageCapture
from .image_processor import ImageProcessor, ImageProcessorCallback


def create_camera(params: Camera.Parameters):
    from .opencv_camera import OpenCvCamera
    camera = OpenCvCamera(params)
    if params.threaded:
        from .threaded_camera import ThreadedCamera
        camera = ThreadedCamera(camera)
    return camera


def create_image_processor(params:ImageProcessor.Parameters, capture:ImageCapture,
                            callback: Optional[ImageProcessorCallback]=None) -> ImageProcessor:
    return ImageProcessor(params, capture, callback)