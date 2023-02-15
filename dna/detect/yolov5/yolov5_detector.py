from typing import List, Optional
from pathlib import Path
from urllib.parse import parse_qs

import yaml
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from dna import Box, Frame
from dna.utils import parse_query
from dna.detect import ObjectDetector, Detection

from pathlib import Path
import logging
LOGGER = logging.getLogger(f"dna.detector.{Path(__file__).parent.stem}")


def load(query: str):
    args = parse_query(query)
    return Yolov5Detector(**args)

class Yolov5Detector(ObjectDetector):
    def __init__(self, **kwargs) -> None:
        model_id = 'yolov5' + kwargs.get('model', 's')
        LOGGER.info(f'Loading {Yolov5Detector.__name__}: model={model_id}')

        self.model = torch.hub.load('ultralytics/yolov5', model_id, pretrained=True, verbose=False)
        self.names = self.model.names

        score = kwargs.get('score')
        if score is not None:
            self.model.conf = float(score)    

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