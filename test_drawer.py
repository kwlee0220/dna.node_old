from typing import Tuple, List
from contextlib import closing

import cv2
import numpy as np
from omegaconf import OmegaConf

from dna import Box, Size2d, Point, color, plot_utils
from dna.camera import create_opencv_camera_from_conf
from dna.utils import RectangleDrawer, PolygonDrawer
from dna.zone import Zone

img = None

SHIFT = 50

def create_blank_image(size:Size2d, *, color:color=color.WHITE) -> np.ndarray:
    from dna.color import WHITE, BLACK
    blank_img = np.zeros([size.height, size.width, 3], dtype=np.uint8)
    if color == WHITE:
        blank_img.fill(255)
    elif color == BLACK:
        pass
    else:
        blank_img[:,:,0].fill(color[0])
        blank_img[:,:,1].fill(color[1])
        blank_img[:,:,2].fill(color[2])
    
    return blank_img

camera_conf = OmegaConf.create()
# camera_conf.uri = "data/2022/crops/etri_07_crop.mp4"
camera_conf.uri = "data/2023/etri_07_join.mp4"
# camera_conf.uri = "data/ai_city/ai_city_t3_c01.avi"
# camera_conf.uri = "data/crossroads/crossroad_04.mp4"
# camera_conf.uri = "output/track_07.mp4"
camera_conf.begin_frame = 70
camera = create_opencv_camera_from_conf(camera_conf)

localizer = None
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
# localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json', 0, contact_point=ContactPointType.Simulation)

track_zones = [
    [[182, 399], [182, 478], [523, 503], [799, 460], [1194, 299],
      [1408, 313], [1362, 488], [1807, 595], [1814, 930], [4, 927], [0, 399]]
]
blind_zones = [
]
exit_zones = [
    [[55, 492], [7, 505], [7, 441], [103, 440], [100, 459]],
    [175, 395, 257, 485],
    [1148, 200, 1415, 310],
    [600, 930, 1918, 1080],
]
zones = [
    [[93, 482], [328, 667], [-25, 812], [-17, 519], [93, 482]],
    # [[686, 481], [1242, 584], [1407, 320], [1154, 315], [686, 481]],
    # [[209, 975], [356, 806], [1035, 631], [1598, 645], [1644, 980], [209, 975]],
]

with closing(camera.open()) as cap:
    src_img = cap().image
    
    box = Box.from_image(src_img)
    
    img = create_blank_image(box.expand(50).size(), color=color.BLACK)
    roi = box.translate([SHIFT, SHIFT])
    roi.update_roi(img, src_img)
# img = cv2.imread("output/2023/etri_06_trajs.jpg", cv2.IMREAD_COLOR)
# img = cv2.imread("output/ETRI_221011.png", cv2.IMREAD_COLOR)

def shift(coords, amount=SHIFT):
    if isinstance(coords[0], list):
        return [[c[0]+amount, c[1]+amount] for c in coords]
    else:
        return [c+amount for c in coords]

for coords in track_zones:
    img = plot_utils.draw_polygon(img, shift(coords), color.GREEN, 1)
for coords in exit_zones:
    img = Zone.from_coords(shift(coords)).draw(img, color.ORANGE, line_thickness=2)
for coords in blind_zones:
    img = Zone.from_coords(shift(coords)).draw(img, color.BLUE, line_thickness=2)
for coords in zones:
    img = Zone.from_coords(shift(coords), as_line_string=True).draw(img, color.RED, line_thickness=1)

def image_to_world(localizer:WorldCoordinateLocalizer, pt_p):
    pt_m = localizer.from_image_coord(pt_p)
    return localizer.to_world_coord(pt_m).astype(int)

polygon = []
polygon = [[111, 496], [328, 667], [-25, 812], [-17, 530], [64, 503]]
coords = PolygonDrawer(img, shift(polygon)).run()
if localizer:
    coords = [list(image_to_world(localizer, coord)) for coord in coords]

print(shift(coords, -SHIFT))

cv2.destroyAllWindows()