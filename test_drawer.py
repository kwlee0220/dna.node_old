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
camera_conf.uri = "data/2022/etri_051.mp4"
# camera_conf.uri = "data/ai_city/ai_city_10.mp4"
# camera_conf.uri = "data/crossroads/crossroad_04.mp4"
# camera_conf.uri = "output/track_07.mp4"
camera_conf.begin_frame = 100
camera:Camera = create_camera_from_conf(camera_conf)

track_zones = [
    [[1, 624], [674, 367], [713, 322], [670, 293], [676, 195], [1026, 240], [1357, 311], [1785, 455], [1764, 562],
      [1420, 534], [1224, 839], [1123, 1077], [1, 1078]]
]
blind_zones = [
]
exit_zones = [
    [553, 177, 703, 316],
    [1700, 388, 1900, 570],
    [[103, 521], [347, 822], [372, 1078], [284, 1079], [0, 1080], [2, 521]]
]
zones = [
    # [[153, 387], [206, 518], [1176, 540]],
    # [[1326, 894], [1498, 517]],
    # [[81, 740], [1349, 740]],
    # [[103, 348], [143, 728]],
    # [[1562, 517], [1376, 928], [1466, 1049]],
    # [[338, 548], [1254, 953]],
    # [[34, 407], [228, 401], [449, 540]],
    # [[35, 393], [298, 380], [468, 326]],
    # [[735, 212], [710, 281], [463, 363]],
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