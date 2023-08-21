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
# camera_conf.uri = "data/2023/etri_07_join.mp4"
# camera_conf.uri = "data/ai_city/ai_city_t3_c01.avi"
# camera_conf.uri = "data/crossroads/crossroad_04.mp4"
camera_conf.uri = "data/shibuya_7_8.mp4"
# camera_conf.uri = "output/track_07.mp4"
camera_conf.begin_frame = 6701
camera = create_opencv_camera_from_conf(camera_conf)

localizer = None
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
# localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json', 0, contact_point=ContactPointType.Simulation)

track_zones = [
    [[575, 178], [690, 180], [898, 232], [857, 299], [912, 397], [729, 565], [255, 525], [114, 321], [253, 224]]
]
blind_zones = [
]
exit_zones = [
    [[491, 155], [574, 192], [250, 238], [228, 192]],   # 윗쪽 출구
    [[924, 200], [770, 152], [707, 184], [708, 217], [866, 263]],   # 오른쪽 출구
    [[588, 540], [867, 349], [942, 387], [748, 574]],   # 아랫쪽 출구
    [[113, 319], [172, 284], [386, 511], [259, 533]],   # 왼쪽 출구
]
zones = [
    [[573, 172], [580, 209], [269, 264], [242, 237]],   # 위쪽 라인
    [[863, 276], [844, 300], [626, 202], [643, 177]],   # 오른쪽 라인
    [[856, 323], [510, 531]],   # 아랫쪽 라인
    [[398, 532], [445, 481], [239, 266], [178, 264], [120, 317], [257, 528], [398, 532]],   # 왼쪽 라인
    # [[887, 287], [886, 362], [251, 370], [253, 289], [887, 287]],   # C-0
    # [[888, 368], [888, 427], [249, 437], [250, 374], [888, 368]],   # 0-C
    # [[168, 167], [170, 264], [106, 265], [92, 167], [168, 167]],    # 0-B
    # [[94, 299], [35, 299], [36, 227], [95, 276], [94, 299]],        # B-0
    # [[90, 959], [32, 958], [35, 474], [93, 473], [90, 959]],        # 0-A
    # [[156, 957], [100, 958], [101, 530], [160, 529], [156, 957]],   # A-0
    # [[579, 1108], [200, 1107], [201, 1051], [578, 1056], [577, 1108]],  # D-1
    # [[577, 1173], [196, 1173], [199, 1111], [580, 1112], [577, 1173]],  # 1-D
    
    # [[19, 1233], [20, 1111], [81, 1131], [80, 1236], [19, 1233]],    # 1-E
    # [[86, 1236], [86, 1145], [160, 1177], [157, 1238], [86, 1236]],  # E-1
    # [[39, 310], [235, 308], [233, 455], [39, 455], [39, 310]],      # 0
    # [[30, 995], [161, 995], [176, 1167], [28, 1102],[30, 995]],     # 1
    # [250, 230, 820, 500]
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
# polygon = [[268, 263], [600, 204], [574, 171], [238, 237]]
coords = PolygonDrawer(img, shift(polygon)).run()
coords = shift(coords, -SHIFT)
if localizer:
    coords = [list(image_to_world(localizer, coord)) for coord in coords]

print(coords)

cv2.destroyAllWindows()