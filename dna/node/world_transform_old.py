# from typing import List, Union
from typing import Tuple
from collections import namedtuple

from omegaconf import OmegaConf
import numpy as np
import cv2

from dna import Point
from dna.node.track_event import TrackEvent
from dna.node.event_processor import EventProcessor

import logging
LOGGER = logging.getLogger("dna.node.pipeline")


CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos')
class WorldTransform(EventProcessor):
    def __init__(self, conf:OmegaConf) -> None:
        EventProcessor.__init__(self)

        self.satellite, cameras = WorldTransform.load_config_file(conf.camera_geometry)
        camera_params = cameras[conf.get('camera_index', 0)]
        self.geometry = CameraGeometry(camera_params['K'], camera_params['distort'],
                                        camera_params['ori'], camera_params['pos'])
        self.utm_origin = None
        epsg_code = conf.get('epsg_code', 'EPSG:3857')
        if epsg_code == 'EPSG:4326':
            self.utm_origin = np.array([0, 0.])
        else:
            from pyproj import Transformer
            transformer = Transformer.from_crs('EPSG:4326', epsg_code)
            self.utm_origin = transformer.transform(*self.satellite['origin_latlon'])
        self.utm_offset = conf.get('world_coords_offset')
        if self.utm_offset:
            self.utm_offset = np.array(self.utm_offset)

    def handle_event(self, ev:TrackEvent) -> None:
        wcoord, dist = self.to_world_coord(ev)
        wcoord = Point.from_np(wcoord) if wcoord is not None else None
        if wcoord is not None and self.utm_offset is not None:
            wcoord = wcoord + self.utm_offset
        updated = ev.updated(world_coord=wcoord, distance=dist)
        self.publish_event(updated)

    def to_world_coord(self, ev:TrackEvent) -> Tuple[np.ndarray,np.double]:
        # 물체의 boundingbox에서 point를 선택
        tl_x, tl_y, br_x, br_y = ev.location.to_tlbr()
        pt = [(tl_x + br_x) / 2, br_y]
        # pt = [(tl_x + br_x) / 2, (br_y + tl_y) / 2]

        pt_m, dist = self._localize_point(pt)
        if pt_m is None and LOGGER.isEnabledFor(logging.INFO):
            LOGGER.warning(f"invalid track: luid={ev.luid}, point={pt}, frame_index={ev.frame_index}")
            return None, None

        return pt_m[[0,2]] + self.utm_origin, dist

    def _localize_point(self, pt, offset=0.) -> Tuple[np.ndarray,np.double]:
        pt_n = cv2.undistortPoints(np.array(pt, dtype=self.geometry.K.dtype), self.geometry.K,
                                    self.geometry.distort).squeeze(axis=1)
        pt_c = np.matmul(self.geometry.ori, np.append(pt_n, 1))
        if pt_c[1] > 0:
            scale = (offset - self.geometry.pos[1]) / pt_c[1]
            position = scale * pt_c + self.geometry.pos
            distance = scale * np.linalg.norm(pt_c)
            return position, distance
        else:
            return None, None

    @staticmethod
    def load_config_file(filename):
        '''Load satellite and camera data from the given file'''
        if filename.endswith('.json'):
            with open(filename, 'r') as f:
                import json
                json_load = json.load(f)

                satellite = {}
                for key, value in json_load['satellite'].items():
                    if key.endswith('_ndarray'):
                        satellite[key[:-8]] = np.array(value)
                    else:
                        satellite[key] = value
                cameras = []
                for cam in json_load['cameras']:
                    cam_load = {}
                    for key, value in cam.items():
                        if key.endswith('_ndarray'):
                            cam_load[key[:-8]] = np.array(value)
                        else:
                            cam_load[key] = value
                    cameras.append(cam_load)
                return satellite, cameras
        else:
            raise ValueError(f"invalid camera geometry file: {filename}")