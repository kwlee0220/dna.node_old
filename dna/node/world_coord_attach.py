from collections import namedtuple

from omegaconf import OmegaConf

from dna import Point
from dna.node.track_event import TrackEvent
from dna.node.event_processor import EventProcessor

import logging
LOGGER = logging.getLogger("dna.node.pipeline")

CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos,polygons,planes,cylinder_table,cuboid_table')
class WorldCoordinateAttacher(EventProcessor):
    def __init__(self, conf:OmegaConf) -> None:
        EventProcessor.__init__(self)

        from .world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
        epsg_code = conf.get('epsg_code', 'EPSG:5186')
        self.localizer = WorldCoordinateLocalizer(conf.camera_geometry, conf.get('camera_index', 0), epsg_code)
        self.contact_point = conf.get('contact_point', ContactPointType.Simulation.value)
        self.contact_point = ContactPointType(self.contact_point)

    def handle_event(self, ev:TrackEvent) -> None:
        pt = self.localizer.select_contact_point(ev.location.to_tlbr())
        pt_lonlat, dist = self.localizer.to_world_coord(pt)
        if pt_lonlat is not None:
            pt_lonlat = Point.from_np(pt_lonlat)
        updated = ev.updated(world_coord=pt_lonlat, distance=dist)
        self.publish_event(updated)