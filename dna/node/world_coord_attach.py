
from typing import List, Dict, Tuple, Union
from collections import namedtuple

from omegaconf import OmegaConf

from dna import Point
from dna.node import TrackEvent, TimeElapsed
from dna.node.event_processor import EventProcessor

import logging
LOGGER = logging.getLogger("dna.node.world_coord")

CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos,polygons,planes,cylinder_table,cuboid_table')
class WorldCoordinateAttacher(EventProcessor):
    def __init__(self, conf:OmegaConf) -> None:
        EventProcessor.__init__(self)

        from .world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
        camera_index = conf.get('camera_index', 0)
        epsg_code = conf.get('epsg_code', 'EPSG:5186')
        self.localizer = WorldCoordinateLocalizer(conf.camera_geometry, camera_index, epsg_code)
        self.contact_point = conf.get('contact_point', ContactPointType.Simulation.value)
        self.contact_point = ContactPointType(self.contact_point)
        
        self.logger = LOGGER
        self.logger.info((f'created: WorldCoordinateAttacher: camera_index={camera_index}, '
                          f'epsg_code={epsg_code}, '
                          f'contact_point={self.contact_point}'))

    def handle_event(self, ev:Union[TrackEvent, TimeElapsed]) -> None:
        if isinstance(ev, TrackEvent):
            pt_m, dist = self.localizer.from_camera_box(ev.location.tlbr)
            world_coord = self.localizer.to_world_coord(pt_m) if pt_m is not None else None
            if world_coord is not None:
                world_coord = Point.from_np(world_coord)
            updated = ev.updated(world_coord=world_coord, distance=dist)
            self._publish_event(updated)
        else:
            self._publish_event(ev)