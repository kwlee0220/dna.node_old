from __future__ import annotations
from typing import Optional

from contextlib import ExitStack, closing, contextmanager
        

@contextmanager
def multi_camera_context(camera_list):
    with ExitStack() as stack:
        yield [stack.enter_context(closing(camera.open())) for camera in camera_list]