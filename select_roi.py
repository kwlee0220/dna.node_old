from typing import Tuple, List
from contextlib import closing

import cv2
import numpy as np
from omegaconf import OmegaConf

from dna import Box, color
from dna.support import plot_utils
from dna.support.rectangle_drawer import RectangleDrawer
from dna.camera import Camera, ImageProcessor, create_opencv_camera_from_conf
from dna.support.polygon_drawer import PolygonDrawer

img = None

camera_conf = OmegaConf.create()
camera_conf.uri = "data/crossroads/crossroad_03.mp4"
# camera_conf.uri = "output/result.jpg"
# camera_conf.begin_frame = 2242
camera = create_opencv_camera_from_conf(camera_conf)

with closing(camera.open()) as cap:
    img = cap().image
# img = cv2.imread("output/result.jpg", cv2.IMREAD_COLOR)

x, y, w, h = cv2.selectROI(windowName="select ROI", img=img)
print(x, y, x+w, y+h)

cv2.destroyAllWindows()