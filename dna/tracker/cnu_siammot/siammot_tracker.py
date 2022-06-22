from datetime import datetime
from typing import List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import sys
from enum import Enum

import numpy as np
import cv2
from omegaconf.omegaconf import OmegaConf
import logging

from dna import Frame

FILE = Path(__file__).absolute()
DEEPSORT_DIR = str(FILE.parents[0] / 'deepsort')
if not DEEPSORT_DIR in sys.path:
    sys.path.append(DEEPSORT_DIR)

import dna
from dna import Box
from dna.tracker import Track, TrackState

from detectron2.config import get_cfg
from detectron2.structures import Instances

from .siammot.engine.predictor import CustomPredictor
from .siammot.config import add_siammot_config
from .centernet.config import add_centernet_config

DEFAULT_START_SCORE = 0.8
DEFAULT_TRACK_THRESH = 0.48
DEFAULT_RESUME_TRACK_THRESH = 0.5
DEFAULT_COSINE_WINDOW_WEIGHT = 0.2
DEFAULT_DETECTION_SCORE_THRESH = 0.8
DEFAULT_TRACK_POOLER_SCALES = (0.125,)
DEFAULT_MAX_DORMANT_FRAMES = 10

_CONFIG_CENTERNET_URI = 'https://drive.google.com/u/0/uc?id=10hIrYUW0XvuWtj197Q1xXJp4ca8-1KDf'

class SiamMOT():
    def __init__(self, models, config_path, device="cuda:0"):
        super(SiamMOT).__init__()
        cfg = self.setup_detectron_config(config_path, device)
        cfg.MODEL.WEIGHTS = models
        self.tracker = CustomPredictor(cfg)

    def setup_detectron_config(self, config_path, device="cuda:0"):
        cfg = get_cfg()
        add_siammot_config(cfg)
        add_centernet_config(cfg)
        cfg.merge_from_file(config_path)

        cfg.DATASETS.TRAIN = ('bdd100k_train',)
        cfg.DATASETS.TEST = ('bdd100k_val',)

        cfg.MODEL.DEVICE = device
        cfg.MODEL.META_ARCHITECTURE = "MOT_RCNN"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
        cfg.MODEL.TRACK_ON = True
        cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False

        cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH = DEFAULT_START_SCORE
        cfg.MODEL.TRACK_HEAD.TRACK_THRESH = DEFAULT_TRACK_THRESH
        cfg.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH = DEFAULT_RESUME_TRACK_THRESH
        cfg.MODEL.TRACK_HEAD.EMM.COSINE_WINDOW_WEIGHT = DEFAULT_COSINE_WINDOW_WEIGHT

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DEFAULT_DETECTION_SCORE_THRESH
        cfg.MODEL.TRACK_HEAD.POOLER_SCALES = DEFAULT_TRACK_POOLER_SCALES
        cfg.MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES = DEFAULT_MAX_DORMANT_FRAMES

        return cfg

    def track(self, frame: Frame) -> List[Track]:
        result = self.tracker(frame)
        track = result['instances'].to("cpu")
        return self.to_dna_track(track, frame.index, frame.ts)

    def to_dna_track(self, track: Instances, frame_idx: int, ts:float):
        track = track[track.ids != -1]
        ids = track.ids.numpy()
        bboxes = track.pred_boxes.tensor.numpy()
        labels = track.pred_classes.numpy()
        scores = track.scores.numpy()
        track_state = track.state.numpy()

        result = []
        for idx, track_id in enumerate(ids):
            if track_state[idx] >= -1:
                state = TrackState(2)
            if track_state[idx] <= -2:
                state = TrackState(4)

            result.append(Track(id=int(track_id),
                                state=state,
                                location=Box(bboxes[idx]),
                                frame_index=frame_idx,
                                ts=ts))
        return result