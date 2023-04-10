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
camera_conf.uri = "data/2023/etri_07_join.mp4"
# camera_conf.uri = "data/ai_city/ai_city_t3_c01.avi"
# camera_conf.uri = "data/crossroads/crossroad_04.mp4"
# camera_conf.uri = "output/track_07.mp4"
camera_conf.begin_frame = 70
camera:Camera = create_camera_from_conf(camera_conf)

localizer = None
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
# localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json', 0, contact_point=ContactPointType.Simulation)

track_zones = [
    [[268, 485], [669, 450], [1194, 299], [1408, 313], [1362, 488], [1807, 595], [1814, 930], [4, 927], [0, 698]]
]
blind_zones = [
]
exit_zones = [
    [[0, 704], [271, 485], [203, 445], [2, 587]],
    [1148, 200, 1415, 310],
    [600, 930, 1918, 1080],
]
zones = [
    # [[8, 613], [215, 509], [661, 507], [143, 840], [8, 613]],
    # [[686, 481], [1242, 584], [1407, 320], [1154, 315], [686, 481]],
    # [[209, 975], [356, 806], [1035, 631], [1598, 645], [1644, 980], [209, 975]],
]

with closing(camera.open()) as cap:
    img = cap().image
# img = cv2.imread("output/2023/etri_06_trajs.jpg", cv2.IMREAD_COLOR)
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
polygon = [[268, 485], [523, 503], [799, 460], [1194, 299], [1408, 313], [1362, 488], [1807, 595], [1814, 930], [4, 927], [0, 698]]
coords = PolygonDrawer(img, polygon).run()
if localizer:
    coords = [list(image_to_world(localizer, coord)) for coord in coords]

print(coords)

cv2.destroyAllWindows()