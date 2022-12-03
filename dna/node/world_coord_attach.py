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

        from .world_coord_localizer import WorldCoordinateLocalizer
        self.localizer = WorldCoordinateLocalizer(conf.camera_geometry, conf.get('camera_index', 0))

    def handle_event(self, ev:TrackEvent) -> None:
        pt = self.localizer.select_contact_point(ev.location.to_tlbr())
        pt_lonlat, dist = self.localizer.to_lonlat_coord(pt)
        if pt_lonlat is not None:
            pt_lonlat = Point.from_np(pt_lonlat)
        updated = ev.updated(lonlat_coord=pt_lonlat, distance=dist)
        self.publish_event(updated)