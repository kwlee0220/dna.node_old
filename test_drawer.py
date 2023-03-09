from typing import Tuple, List
from contextlib import closing

import cv2
import numpy as np
from omegaconf import OmegaConf

from dna import Box, color, plot_utils
from dna.camera import Camera, ImageProcessor
from dna.camera.utils import create_camera_from_conf
from dna.utils import RectangleDrawer, PolygonDrawer
from dna.zone import Zone

img = None

camera_conf = OmegaConf.create()
# camera_conf.uri = "data/2022/crops/etri_07_crop.mp4"
# camera_conf.uri = "data/2022/etri_051.mp4"
camera_conf.uri = "data/ai_city/ai_city_11.mp4"
# camera_conf.uri = "data/crossroads/crossroad_04.mp4"
# camera_conf.uri = "output/track_07.mp4"
camera_conf.begin_frame = 100
camera:Camera = create_camera_from_conf(camera_conf)

track_zones = [
    [[2,228], [1916,300], [1916,1076], [2,1074]]
]
blind_zones = [
    [0, 800, 1920, 1080],
    [1527, 441, 1917, 1075]
]
exit_zones = [
    [[981, 267], [1492, 284], [1230, 216], [871, 216]],
    [0, 800, 1920, 1080],
    [0, 396, 85, 725],
    [1525, 440, 1920, 1080]
]
zones = [
    [[153, 387], [206, 518], [1176, 540]],
    [[1326, 894], [1498, 517]],
    [[81, 740], [1349, 740]],
    [[91, 693], [90, 344]]
]

with closing(camera.open()) as cap:
    img = cap().image
# img = cv2.imread("conf/ai_city/cam_4.jpg", cv2.IMREAD_COLOR)

for coords in track_zones:
    img = plot_utils.draw_polygon(img, coords, color.GREEN, 2)
for coords in exit_zones:
    img = Zone.from_coords(coords).draw(img, color.ORANGE, line_thickness=2)
for coords in blind_zones:
    img = Zone.from_coords(coords).draw(img, color.BLUE, line_thickness=2)
for coords in zones:
    img = Zone.from_coords(coords, as_line_string=True).draw(img, color.RED, line_thickness=3)

polygon = []
# polygon = [[1406, 282], [1768, 294], [1643, 236], [1334, 229]]
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