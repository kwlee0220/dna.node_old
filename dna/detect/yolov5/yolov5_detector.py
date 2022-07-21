from typing import List, Optional
from pathlib import Path
from urllib.parse import parse_qs
import logging

import yaml
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from dna import get_logger, Box, Frame
from dna.utils import parse_query
from dna.detect import ObjectDetector, Detection, LOGGER


def load(query: str):
    args = parse_query(query)
    model_id = 'yolov5' + args.get('model', 's')

    model = torch.hub.load('ultralytics/yolov5', model_id, pretrained=True, verbose=False)

    score = args.get('score')
    if score is not None:
        model.conf = float(score)

    LOGGER.info(f'Loading Yolov5Detector: model={model_id}')
    return Yolov5Detector(model)


class Yolov5Detector(ObjectDetector):
    def __init__(self, model) -> None:
        self.model = model
        self.names = model.names

    @torch.no_grad()
    def detect(self, frame: Frame) -> List[Detection]:
        # Convert
        img = frame.image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # inference
        preds = self.model(img).xyxy[0].cpu().numpy()

        det_list = []
        for i, pred in enumerate(preds):  # detections per image
            bbox = Box.from_tlbr(pred[:4])
            name = self.names[int(pred[5])]
            confi = pred[4]
            det_list.append(Detection(bbox, name, confi))

        return det_list