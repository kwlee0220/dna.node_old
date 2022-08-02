
from typing import Tuple
from contextlib import closing

import cv2
import numpy as np

from dna import Box
from dna.camera.opencv_camera import OpenCvCamera, OpenCvVideFile
from dna.utils import RectangleDrawer


img = None
uri = "data/etri/etri_053.mp4"
camera = OpenCvVideFile(uri) if OpenCvCamera.is_video_file(uri) else OpenCvCamera(uri)
with closing(camera.open()) as cap:
    img = cap().image

image, box = RectangleDrawer(img).run()
print(box.top_left(), box.bottom_right())

cv2.destroyAllWindows()    