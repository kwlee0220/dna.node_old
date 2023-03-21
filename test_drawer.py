from typing import Tuple, List
from contextlib import closing

import cv2
import numpy as np
from omegaconf import OmegaConf

from dna import Box, Point, color, plot_utils
from dna.camera import Camera, ImageProcessor
from dna.camera.utils import create_camera_from_conf
from dna.utils import RectangleDrawer, PolygonDrawer
from dna.zone import Zone

img = None

camera_conf = OmegaConf.create()
# camera_conf.uri = "data/2022/crops/etri_07_crop.mp4"
camera_conf.uri = "data/2023/etri_06_join.mp4"
# camera_conf.uri = "data/ai_city/ai_city_t3_c01.avi"
# camera_conf.uri = "data/crossroads/crossroad_04.mp4"
# camera_conf.uri = "output/track_07.mp4"
camera_conf.begin_frame = 83
camera:Camera = create_camera_from_conf(camera_conf)

localizer = None
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
# localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json', 0, contact_point=ContactPointType.Simulation)

track_zones = [
    [[205, 607], [792, 442], [941, 332], [1099, 332], [1297, 496], [1826, 607], [1829, 1001], [6, 853], [4, 606]]
]
blind_zones = [
]
exit_zones = [
    [1017, 275, 1111, 334],
    [1600, 686, 1920, 1080],
    [3, 600, 120, 900],
]
zones = [
    [[1409, 516], [955, 721], [974, 1037], [1836, 1033], [1827, 606], [1409, 516]],
    [[519, 521], [700, 714], [696, 1047], [20, 1051], [18, 562], [519, 521]],
    [[651, 529], [1352, 526], [1122, 353], [896, 354], [651, 529]],
]

with closing(camera.open()) as cap:
    img = cap().image
img = cv2.imread("output/2023/etri_06_trajs.jpg", cv2.IMREAD_COLOR)
# img = cv2.imread("output/ETRI_221011.png", cv2.IMREAD_COLOR)

for coords in track_zones:
    img = plot_utils.draw_polygon(img, coords, color.GREEN, 1)
for coords in exit_zones:
    img = Zone.from_coords(coords).draw(img, color.ORANGE, line_thickness=2)
for coords in blind_zones:
    img = Zone.from_coords(coords).draw(img, color.BLUE, line_thickness=2)
for coords in zones:
    img = Zone.from_coords(coords, as_line_string=True).draw(img, color.RED, line_thickness=1)

def image_to_world(localizer:WorldCoordinateLocalizer, pt_p):
    pt_m = localizer.from_image_coord(pt_p)
    return localizer.to_world_coord(pt_m).astype(int)

polygon = []
# polygon = [[182, 399], [182, 478], [523, 503], [799, 460], [1219, 288], [1221, 265],
#       [1420, 265], [1362, 488], [1807, 595], [1814, 930], [4, 927], [0, 399]]
coords = PolygonDrawer(img, polygon).run()
if localizer:
    coords = [list(image_to_world(localizer, coord)) for coord in coords]

print(coords)

cv2.destroyAllWindows()