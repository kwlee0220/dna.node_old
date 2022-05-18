from typing import Optional, Dict
from argparse import Namespace
from numpy import isin
from omegaconf import OmegaConf

import dna
from dna import Size2d
from dna.camera import ImageCapture, ImageProcessor, ImageProcessorCallback

def create_image_processor(args: OmegaConf, capture: ImageCapture,
                            callback: Optional[ImageProcessorCallback]=None) -> ImageProcessor:
    if capture is None:
        capture = dna.create_camera(args).open()

    output_video = args.output_video if hasattr(args, "output_video") else None
    show_progress = args.show_progress if hasattr(args, "show_progress") else None
    return ImageProcessor(capture, callback,
                            window_name=args.window_name,
                            output_video=output_video,
                            show_progress=show_progress,
                            pause_on_eos=False)

def execute_image_processor(args: OmegaConf, capture: Optional[ImageCapture]=None,
                            callback: Optional[ImageProcessorCallback]=None):
    return create_image_processor(args=args, capture=capture, callback=callback).run()