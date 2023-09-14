from contextlib import closing

import cv2
import numpy as np
from omegaconf import OmegaConf

from dna import Box, Size2d, Point, color
from dna.support import plot_utils
from dna.support.rectangle_drawer import RectangleDrawer
from dna.camera import create_opencv_camera_from_conf
from dna.support.polygon_drawer import PolygonDrawer
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
# camera_conf.uri = "data/2022/etri_041.mp4"
# camera_conf.uri = "data/2023/etri_07_join.mp4"
# camera_conf.uri = "data/ai_city/ai_city_t3_c01.avi"
# camera_conf.uri = "data/crossroads/crossroad_04.mp4"
# camera_conf.uri = "data/shibuya_7_8.mp4"
camera_conf.uri = "data/2023/etri_10.mp4"
camera_conf.begin_frame = 10
camera = create_opencv_camera_from_conf(camera_conf)

localizer = None
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
# localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json', 0, contact_point=ContactPointType.Simulation)

track_zones = [
    [[1786, 87], [1723, 135], [1769, 175], [908, 781], [730, 802], [389, 593], [121, 733], [551, 1080], [1121, 1076], [1915, 253], [1916, 121]]
]
blind_zones = [
]
exit_zones = [
    [[1737, 149], [1808, 89], [1786, 76], [1711, 132]],   # 윗쪽 출구
    [[123, 747], [403, 595], [374, 569], [99, 720]],   # 아랫쪽 출구
    [[1245, 970], [1164, 929], [1017, 1078], [1143, 1083]],
]
zones = [
    [[1749, 182], [1857, 89]],   # 위쪽 라인
    [[1722, 197], [1911, 280]],   # 오른쪽 라인
    [[1304, 910], [1171, 907], [1003, 1030], [992, 1102]],
    [[162, 789], [460, 613]],
]

with closing(camera.open()) as cap:
    src_img = cap().image
    # src_img = cv2.imread("output/ETRI_221011.png", cv2.IMREAD_COLOR)
    
    box = Box.from_image(src_img)
    
    img = create_blank_image(box.expand(50).size(), color=color.BLACK)
    roi = box.translate([SHIFT, SHIFT])
    roi.update_roi(img, src_img)
# img = cv2.imread("output/2023/etri_06_trajs.jpg", cv2.IMREAD_COLOR)
# img = cv2.imread("output/ETRI_221011.png", cv2.IMREAD_COLOR)

def shift(coords, amount=SHIFT):
    if not coords:
        return []
    elif isinstance(coords[0], list):
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
# polygon = [[432, 824], [218, 644], [3, 694], [4, 1076], [120, 1072]]
coords = PolygonDrawer(img, shift(polygon)).run()
coords = shift(coords, -SHIFT)
if localizer:
    coords = [list(image_to_world(localizer, coord)) for coord in coords]

print(coords)

cv2.destroyAllWindows()