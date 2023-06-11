from __future__ import annotations

import cv2
import numpy as np

from dna import Box, Image, Point, color
from .running_stabilizer import RunningStabilizer
from .world_coord_localizer import ContactPointType, WorldCoordinateLocalizer


def _xy(pt):
    return pt.xy if isinstance(pt, Point) else pt


class TrajectoryDrawer:
    def __init__(self, box_trajs: dict[str,list[Box]], camera_image: Image, world_image: Image=None,
                localizer:WorldCoordinateLocalizer=None, stabilizer:RunningStabilizer=None, traj_color:color=color.RED) -> None:
        self.box_trajs = box_trajs
        self.localizer = localizer
        self.stabilizer = stabilizer
        self.camera_image = camera_image
        self.world_image = world_image
        self.traj_color = traj_color
        self.thickness = 2

        self.show_world_coords = False
        self.show_stabilized = False

    def draw_to_file(self, outfile:str) -> None:
        convas = self.draw(pause=False)
        cv2.imwrite(outfile, convas)

    def _put_text(self, convas:Image, luid:int=None):
        id_str = f'luid={luid}, ' if luid is not None else ''
        view = "world" if self.show_world_coords else "camera"
        contact = self.localizer.contact_point_type.name if self.localizer else ContactPointType.Centroid.name
        stabilized_flag = f', stabilized({self.stabilizer.smoothing_factor})' if self.show_stabilized else ''
        return cv2.putText(convas, f'{id_str}view={view}, contact={contact}{stabilized_flag}',
                            (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, self.traj_color, 2)

    def draw(self, title='trajectories', pause:bool=True, capture_file='output/capture.png') -> Image:
        bg_image = self.world_image if self.world_image is not None and self.show_world_coords else self.camera_image
        convas = bg_image.copy()
        convas = self._put_text(convas)
        for traj in self.box_trajs.values():
            convas = self._draw_trajectory(convas, traj)

        if pause:
            while True:
                cv2.imshow(title, convas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key != 0xFF:
                    if key == ord('c') and self.localizer is not None:
                        cp_value = self.localizer.contact_point_type.value + 1
                        contact_point_type = ContactPointType(cp_value % len(ContactPointType))
                        self.localizer.contact_point_type = contact_point_type
                    elif key == ord('w') and self.localizer is not None:
                        self.show_world_coords = not self.show_world_coords
                    elif key == ord('t'):
                        self.show_stabilized = not self.show_stabilized

                    bg_image = self.world_image if self.world_image is not None and self.show_world_coords else self.camera_image
                    convas = bg_image.copy()
                    convas = self._put_text(convas)
                    for traj in self.box_trajs.values():
                        convas = self._draw_trajectory(convas, traj)

                if key == ord('s'):
                        out_file = capture_file if capture_file else 'output.png'
                        cv2.imwrite(out_file, convas)
            cv2.destroyWindow(title)
        return convas

    def draw_interactively(self, title='trajectories', capture_file='capture.png'):
        id_list = sorted(self.box_trajs)
        try:
            idx = 0
            while True:
                luid = id_list[idx]
                bg_img = self.world_image if self.world_image is not None and self.show_world_coords else self.camera_image
                convas = bg_img.copy()
                convas = self._put_text(convas, luid)
                convas = self._draw_trajectory(convas, self.box_trajs[luid])

                while True:
                    cv2.imshow(title, convas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        return
                    elif key == ord('n'):
                        idx = min(idx+1, len(id_list)-1)
                        break
                    elif key == ord('p'):
                        idx = max(idx-1, 0)
                        break
                    elif key == ord('c') and self.localizer is not None:
                        cp_value = self.localizer.contact_point_type.value + 1
                        contact_point_type = ContactPointType(cp_value % len(ContactPointType))
                        self.localizer.contact_point_type = contact_point_type
                        break
                    elif key == ord('w') and self.localizer is not None:
                        self.show_world_coords = not self.show_world_coords
                        break
                    elif key == ord('t'):
                        self.show_stabilized = not self.show_stabilized
                        break
                    elif key == ord('s'):
                        out_file = capture_file if capture_file else 'output.png'
                        cv2.imwrite(out_file, convas)
        finally:
            cv2.destroyWindow(title)

    def _draw_trajectory(self, convas:Image, traj: list[Box]) -> Image:
        pts = None
        if self.localizer is not None:
            pts = [self.localizer.select_contact_point(box.tlbr) for box in traj]
        else:
            pts = [box.center() for box in traj]

        if self.show_world_coords:
            def camera_to_image(pt):
                pt, _ = self.localizer.from_camera_coord(pt)
                return self.localizer.to_image_coord(pt)

            pts = [camera_to_image(pt_c) for pt_c in pts]

        if self.show_stabilized:
            pts = self.stabilize(pts)

        pts = [_xy(pt) for pt in pts]
        pts = np.rint(np.array(pts)).astype('int32')
        return cv2.polylines(convas, [pts], False, self.traj_color, self.thickness)

    def stabilize(self, traj:list[Point]) -> list[Point]:
        pts_s = []
        for pt in traj:
            pt_s = self.stabilizer.transform(pt)
            if pt_s is not None:
                pts_s.append(pt_s)
        pts_s.extend(self.stabilizer.get_tail())
        self.stabilizer.reset()

        return pts_s