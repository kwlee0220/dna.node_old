# from typing import List, Union
from typing import Tuple
from collections import namedtuple
import json, pickle

from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import cv2

from dna import Point
from dna.node.track_event import TrackEvent
from dna.node.event_processor import EventProcessor

import logging
LOGGER = logging.getLogger("dna.node.pipeline")


def conv_pixel2meter(pt, origin_pixel, meter_per_pixel):
    '''Convert image position to metric position on the satellite image'''
    x = (pt[0] - origin_pixel[0]) * meter_per_pixel
    y = (origin_pixel[1] - pt[1]) * meter_per_pixel
    z = 0
    if len(pt) > 2:
        z = pt[2]
    if type(pt) is np.ndarray:
        return np.array([x, y, z])
    return [x, y, z]

def conv_camera_config(conf_dir:Path, cameras):
    '''Pre-process the multi-camera configuration'''
    for cam in cameras:
        for key in ['K', 'distort', 'rvec', 'tvec', 'ori', 'pos']:
            if key in cam:
                cam[key] = np.array(cam[key])
        if ('focal' in cam) and ('center' in cam):
            cam['K'] = np.array([[cam['focal'][0], 0, cam['center'][0]], [0, cam['focal'][1], cam['center'][1]], [0, 0, 1]])
        if ('rvec' in cam) and ('tvec' in cam):
            cam['ori'] = Rotation.from_rotvec(cam['rvec']).as_matrix().T
            cam['pos'] = -cam['ori'] @ cam['tvec']
        if 'polygons' in cam:
            cam['polygons'] = {int(key): np.array(value).reshape(-1, 2) for key, value in cam['polygons'].items()}
        else:
            cam['polygons'] = {}
        if 'cylinder_file' in cam:
            with open(conf_dir / cam['cylinder_file'], 'rb') as f:
                cam['cylinder_table'] = pickle.load(f)
        if 'cuboid_file' in cam:
            with open(conf_dir / cam['cuboid_file'], 'rb') as f:
                cam['cuboid_table'] = pickle.load(f)

def conv_satellite_config(satellite):
    '''Pre-process the satellite configuration'''
    for key in ['pts', 'planes']:
        if key in satellite:
            satellite[key] = np.array(satellite[key])
    if 'planes' not in satellite:
        satellite['planes'] = []
    if 'roads' in satellite:
        satellite['roads'] = [np.array(road).reshape(-1, 2) for road in satellite['roads']]
        roads_data = []
        for road in satellite['roads']:
            road_m = np.array([conv_pixel2meter(pt, satellite['origin_pixel'], satellite['meter_per_pixel']) for pt in road])
            road_v = road_m[1:] - road_m[:-1]
            road_n = np.linalg.norm(road_v, axis=1)
            roads_data.append(np.hstack((road_m[:-1], road_v, road_n.reshape(-1, 1))))
        satellite['roads_data'] = np.vstack(roads_data)
    else:
        satellite['roads'] = []
        satellite['roads_data'] = []

def predict_center_from_table(bottom_mid, table, dist_threshold=100):
    '''Predict a foot point using the given lookup table and nearest search'''
    x, y = bottom_mid
    dist = np.fabs(table[:,0] - x) + np.fabs(table[:,1] - y)
    min_idx = np.argmin(dist)
    if dist[min_idx] < dist_threshold:
        return table[min_idx,2:4]
    return np.zeros(2)

_DEFAULT_EPSG = 'EPSG:5186'
CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos,polygons,planes,cylinder_table,cuboid_table')
class WorldTransform(EventProcessor):
    def __init__(self, conf:OmegaConf) -> None:
        EventProcessor.__init__(self)

        self.satellite, cameras = WorldTransform.load_config_file(conf.camera_geometry)
        camera_params = cameras[conf.get('camera_index', 0)]
        self.geometry = CameraGeometry(camera_params['K'], camera_params['distort'],
                                        camera_params['ori'], camera_params['pos'],
                                        camera_params['polygons'], self.satellite['planes'],
                                        camera_params['cylinder_table'], camera_params['cuboid_table'])

        # utm_origin 값 설정
        from pyproj import Transformer
        epsg_code = conf.get('epsg_code', _DEFAULT_EPSG)
        tmp_epsg_code = _DEFAULT_EPSG if epsg_code == 'EPSG:4326' else epsg_code
        transformer = Transformer.from_crs('EPSG:4326', tmp_epsg_code)
        self.utm_origin = transformer.transform(*(self.satellite['origin_lonlat'][::-1]))

        if epsg_code == 'EPSG:4326':
            self.transformer = Transformer.from_crs(tmp_epsg_code, 'EPSG:4326')
        else:
            self.transformer = None
        self.utm_offset = conf.get('world_coords_offset')
        if self.utm_offset:
            self.utm_offset = np.array(self.utm_offset)

    def handle_event(self, ev:TrackEvent) -> None:
        # Zone 영역 내에서의 상대 위치 정보를 계산함.
        lcoord, dist = self.to_zone_coord(ev.location.to_tlbr())
        if lcoord is not None:
            lcoord = lcoord[0:2]
            wcoord = self.to_world_coord(lcoord)
            lcoord = Point.from_np(lcoord)
            wcoord = Point.from_np(wcoord)
            # if self.utm_offset is not None:
            #     wcoord = wcoord + self.utm_offset
        else:
            wcoord = None
        updated = ev.updated(world_coord=wcoord, local_coord=lcoord, distance=dist)
        self.publish_event(updated)
    
    def to_zone_coord(self, tlbr: np.array) -> Tuple[np.ndarray,np.double]:
        # Boundingbox에서 point를 선택
        pt = self._select_contact_point(tlbr)
        return self._localize_point(pt)
        
    def to_world_coord(self, zone_coords:np.array) -> np.ndarray:
        world_coord = zone_coords + self.utm_origin
        if self.transformer:
            world_coord = self.transformer.transform(*world_coord)[::-1]
        return world_coord

    def _select_contact_point(self, tlbr:np.ndarray) -> np.ndarray:
        '''Get the bottom middle point of the given bounding box'''
        tl_x, tl_y, br_x, br_y = tlbr
        pt = np.array([(tl_x + br_x) / 2, br_y])

        # delta = predict_center_from_table(pt, self.camera_params['cylinder_table'])
        delta = predict_center_from_table(pt, self.geometry.cuboid_table)
        pt += delta

        return pt

    def _check_polygons(self, pt, polygons):
        '''Check whether the given point belongs to polygons (index) or not (-1)'''
        if len(polygons) > 0:
            for idx, polygon in polygons.items():
                if cv2.pointPolygonTest(polygon, np.array(pt, dtype=np.float32), False) >= 0:
                    return idx
        return -1

    def _get_bbox_bottom_mid(bbox):
        '''Get the bottom middle point of the given bounding box'''
        tl_x, tl_y, br_x, br_y = bbox
        return np.array([(tl_x + br_x) / 2, br_y])
        
    def _localize_point(self, pt) -> Tuple[np.ndarray,np.double]:
        '''Calculate 3D location (unit: [meter]) of the given point (unit: [pixel]) with the given camera configuration'''
        # Make a ray aligned to the world coordinate
        pt_n = cv2.undistortPoints(np.array(pt, dtype=self.geometry.K.dtype), self.geometry.K, self.geometry.distort).flatten()
        r = self.geometry.ori @ np.append(pt_n, 1) # A ray with respect to the world coordinate
        scale = np.linalg.norm(r)
        r = r / scale

        # Get a plane if 'pt' exists inside of any 'polygons'
        n, d = np.array([0, 0, 1]), 0
        plane_idx = self._check_polygons(pt, self.geometry.polygons)
        if (plane_idx >= 0) and (plane_idx < len(self.geometry.planes)):
            n, d = self.geometry.planes[plane_idx][0:3], self.geometry.planes[plane_idx][-1]

        # Calculate distance and position on the plane
        denom = n.T @ r
        if np.fabs(denom) < 1e-6: # If the ray 'r' is almost orthogonal to the plane norm 'n' (~ almost parallel to the plane)
            return None, None
        distance = -(n.T @ self.geometry.pos + d) / denom
        r_c = self.geometry.ori.T @ (np.sign(distance) * r)
        if r_c[-1] <= 0: # If the ray 'r' stretches in the negative direction (negative Z)
            return None, None
        position = self.geometry.pos + distance * r
        return position, np.fabs(distance)

    @staticmethod
    def load_config_file(json_file):
        '''Load the satellite and multi-camera configuration together from a JSON file'''
        conf_dir = Path(json_file).parent
        with open(json_file, 'r') as f:
            config = json.load(f)
            if ('satellite' in config) and ('cameras' in config):
                conv_satellite_config(config['satellite'])
                conv_camera_config(conf_dir, config['cameras'])
                return config['satellite'], config['cameras']
            
    # @staticmethod
    # def load_config_file(filename):
    #     '''Load satellite and camera data from the given file'''
    #     if filename.endswith('.json'):
    #         with open(filename, 'r') as f:
    #             import json
    #             json_load = json.load(f)

    #             satellite = {}
    #             for key, value in json_load['satellite'].items():
    #                 if key.endswith('_ndarray'):
    #                     satellite[key[:-8]] = np.array(value)
    #                 else:
    #                     satellite[key] = value
    #             cameras = []
    #             for cam in json_load['cameras']:
    #                 cam_load = {}
    #                 for key, value in cam.items():
    #                     if key.endswith('_ndarray'):
    #                         cam_load[key[:-8]] = np.array(value)
    #                     else:
    #                         cam_load[key] = value
    #                 cameras.append(cam_load)
    #             return satellite, cameras
    #     else:
    #         raise ValueError(f"invalid camera geometry file: {filename}")