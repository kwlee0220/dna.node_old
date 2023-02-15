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

coords_list = [
    # [[186, 631], [738, 492], [915, 352], [1016, 351], [1017, 334], [1101, 334], [1311, 488], [1726, 587],
    #     [1759, 843], [1733, 1076], [1921, 1075], [1920, 2], [0, -2], [-2, 636],  [1, 1076], [226, 1077], [174, 826]],
    # [[1339, 563], [1296, 1045], [1410, 1054], [1466, 595]],
    # [[1017, 356], [998, 273], [1091, 272], [1097, 355]],
    # [[186, 631], [738, 492], [917, 349], [1112, 347], [1311, 488], [1726, 587], [1759, 843], [1733, 1076],
    #     [1921, 1075], [1920, 2], [0, -2], [-2, 636], [1, 1076], [226, 1077], [174, 826]]
]

with closing(camera.open()) as cap:
    img = cap().image
# img = cv2.imread("output/result.jpg", cv2.IMREAD_COLOR)
for coords in coords_list:
    img = plot_utils.draw_polygon(img, coords, color.ORANGE, 2)

polygon = []
polygon = [[120, 632], [792, 442], [936, 287], [1051, 286], [1297, 496], [1782, 632], [1892, 1076], [1921, 1075],
      [1920, 2], [0, -2], [1, 1076], [276, 1079], [258, 878], [147, 840]]
coords = PolygonDrawer(img, polygon).run()
print(coords)

cv2.destroyAllWindows() 

# 04
# [[952, 382], [1132, 408], [1115, 438], [920, 409]]
# [[1167, 347], [1163, 396], [1143, 391], [1146, 342]]
# [[975, 330], [955, 362], [984, 367], [1001, 333]]

# 05
# [[649, 394], [547, 440], [1107, 749], [1180, 649]]
# [[1441, 354], [1419, 486], [1485, 506], [1501, 378]]
# [[814, 215], [724, 302], [770, 329], [885, 228]]

# 06
# [[1152, 420], [871, 411], [860, 463], [1168, 459]]
# [[490, 574], [451, 1021], [570, 1026], [601, 561]]
# [[1421, 542], [1422, 1048], [1584, 1045], [1538, 560]]

# 07
# [[276, 517], [69, 662], [132, 704], [313, 531]]
# [[1339, 563], [1296, 1045], [1410, 1054], [1466, 595]]
# [[1068, 348], [1028, 381], [1275, 491], [1305, 398]]