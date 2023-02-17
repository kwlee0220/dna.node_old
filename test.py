import cv2
import numpy as np

import torch
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms


transform = transforms.ToTensor()

img = cv2.imread("data/grace_hopper_517x606.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = transform(img)

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()

preprocess = weights.transforms()

batch = [preprocess(img)]
prediction = model(batch)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
print(labels)
print(prediction["boxes"].cpu().detach().numpy())
print(prediction["scores"].cpu().detach().numpy())


# box = draw_bounding_boxes(img, boxes=prediction["boxes"], labels=labels, colors="red", width=4, font_size=30)
# im = to_pil_image(box.detach())
# im.show()