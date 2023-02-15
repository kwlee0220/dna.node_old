from typing import Tuple, List
from contextlib import closing

import cv2
import numpy as np
from omegaconf import OmegaConf

from dna import Box, color, plot_utils
from dna.camera import Camera, ImageProcessor
from dna.camera.utils import create_camera_from_conf
from dna.utils import RectangleDrawer, PolygonDrawer

img = None

camera_conf = OmegaConf.create()
camera_conf.uri = "data/2022/crops/etri_06_crop.mp4"
# camera_conf.uri = "output/result.jpg"
camera_conf.begin_frame = 2242
camera:Camera = create_camera_from_conf(camera_conf)

with closing(camera.open()) as cap:
    img = cap().image
# img = cv2.imread("output/result.jpg", cv2.IMREAD_COLOR)

x, y, w, h = cv2.selectROI(windowName="select ROI", img=img)
print(x, y, x+w, y+h)

cv2.destroyAllWindows()