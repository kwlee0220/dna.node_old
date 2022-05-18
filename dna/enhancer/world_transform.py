# from typing import List, Union
from collections import namedtuple

import pickle

from omegaconf import OmegaConf
from pubsub import Queue
import numpy as np
import cv2

from dna import Point
from dna.node.track_event import TrackEvent
from dna.node.event_processor import EventProcessor
from dna.node.utils import EventPublisher


CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos')

class WorldTransform(EventProcessor):
    def __init__(self, in_queue: Queue, publisher: EventPublisher, conf:OmegaConf) -> None:
        super().__init__(in_queue, publisher)

        with open(conf.file, 'rb') as f:
            self.geometry = pickle.load(f)

    def handle_event(self, ev:TrackEvent) -> None:
        wcoord, _, dist = self._localize_bbox(ev.location.to_tlbr())

        updated = ev.updated(world_coord=Point(wcoord[0], wcoord[2]), distance=dist)
        self.publish_event(updated)

    def _localize_bbox(self, tlbr, offset=0):
        tl_x, tl_y, br_x, br_y = tlbr
        foot_p = [(tl_x + br_x) / 2, br_y]
        head_p = [(tl_x + br_x) / 2, tl_y]

        foot_n, head_n = cv2.undistortPoints(np.array([foot_p, head_p]), self.geometry.K,
                                            self.geometry.distort).squeeze(axis=1)
        foot_c = np.matmul(self.geometry.ori, np.append(foot_n, 1))
        head_c = np.matmul(self.geometry.ori, np.append(head_n, 1))

        scale = (offset - self.geometry.pos[1]) / foot_c[1]
        position = scale * foot_c + self.geometry.pos
        height   = scale * (foot_c[1] - head_c[1])
        distance = scale * np.linalg.norm(foot_c)
        return (position, height, distance)