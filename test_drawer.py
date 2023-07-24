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
# camera_conf.uri = "data/2022/crops/etri_07_crop.mp4"
camera_conf.uri = "data/2023/etri_07_join.mp4"
# camera_conf.uri = "data/ai_city/ai_city_t3_c01.avi"
# camera_conf.uri = "data/crossroads/crossroad_04.mp4"
# camera_conf.uri = "output/track_07.mp4"
camera_conf.begin_frame = 70
camera = create_opencv_camera_from_conf(camera_conf)

localizer = None
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json', 0, contact_point=ContactPointType.Simulation)

track_zones = [
    # [[182, 399], [182, 478], [523, 503], [799, 460], [1194, 299],
    #   [1408, 313], [1362, 488], [1807, 595], [1814, 930], [4, 927], [0, 399]]
]
blind_zones = [
]
exit_zones = [
    # [[55, 492], [7, 505], [7, 441], [103, 440], [100, 459]],
    # [175, 395, 257, 485],
    # [1148, 200, 1415, 310],
    # [600, 930, 1918, 1080],
]
zones = [
    [[887, 287], [886, 362], [251, 370], [253, 289], [887, 287]],   # C-0
    [[888, 368], [888, 427], [249, 437], [250, 374], [888, 368]],   # 0-C
    [[168, 167], [170, 264], [106, 265], [92, 167], [168, 167]],    # 0-B
    [[94, 299], [35, 299], [36, 227], [95, 276], [94, 299]],        # B-0
    [[90, 959], [32, 958], [35, 474], [93, 473], [90, 959]],        # 0-A
    [[156, 957], [100, 958], [101, 530], [160, 529], [156, 957]],   # A-0
    [[579, 1108], [200, 1107], [201, 1051], [578, 1056], [577, 1108]],  # D-1
    [[577, 1173], [196, 1173], [199, 1111], [580, 1112], [577, 1173]],  # 1-D
    
    [[19, 1233], [20, 1111], [81, 1131], [80, 1236], [19, 1233]],    # 1-E
    [[86, 1236], [86, 1145], [160, 1177], [157, 1238], [86, 1236]],  # E-1
    [[39, 310], [235, 308], [233, 455], [39, 455], [39, 310]],      # 0
    [[30, 995], [161, 995], [176, 1167], [28, 1102],[30, 995]],     # 1
]

with closing(camera.open()) as cap:
    src_img = cap().image
    src_img = cv2.imread("output/ETRI_221011.png", cv2.IMREAD_COLOR)
    
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
polygon = [[30, 995], [161, 995], [176, 1167], [28, 1102],[30, 995]]
coords = PolygonDrawer(img, shift(polygon)).run()
coords = shift(coords, -SHIFT)
if localizer:
    coords = [list(image_to_world(localizer, coord)) for coord in coords]

print(coords)

cv2.destroyAllWindows()