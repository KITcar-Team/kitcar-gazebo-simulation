import functools
import math
from typing import List, Tuple

import numpy as np
from gazebo_simulation.msg import CarState as CarStateMsg
from simulation_groundtruth.msg import Section as SectionMsg

from simulation.src.simulation_evaluation.src.speaker.speakers.speaker import Speaker
from simulation.utils.geometry import Polygon, Vector

from .bounding_box import BoundingBox


class LabelSpeaker(Speaker):
    """Speaker that allows to retrieve visible groundtruth objects and their position."""

    def listen(self, msg: CarStateMsg):
        super().listen(msg)
        self.camera_fov = Polygon(msg.view_cone)

        LabelSpeaker._get_visible_sections.cache_clear()

    @functools.lru_cache(1)
    def _get_visible_sections(self) -> List[SectionMsg]:

        if not hasattr(self, "camera_fov"):
            return []

        visibles = []
        for sec in self.sections:
            line_tuple = self.get_road_lines(sec.id)
            left_line, right_line = line_tuple.left, line_tuple.right

            if right_line.intersects(self.camera_fov) or left_line.intersects(
                self.camera_fov
            ):
                visibles.append(sec)

        return visibles

    def _get_visible_obstacles(self) -> List[BoundingBox]:

        obstacles = []
        for sec in self._get_visible_sections():
            for obs, height in self.get_obstacles_in_section(sec.id):
                if obs.intersects(self.camera_fov):
                    points = obs.get_points()
                    points += [p + Vector(0, 0, height) for p in points]
                    obstacles.append(
                        BoundingBox(
                            world_points=points, class_id=-1, class_description="obstacle"
                        )
                    )
        return obstacles

    def _get_visible_surface_markings(self) -> List[BoundingBox]:

        sms = []
        for sec in self._get_visible_sections():
            for type_, sm in self.get_surface_markings_in_section(sec.id):
                if sm.intersects(self.camera_fov):
                    points = sm.get_points()
                    sms.append(
                        BoundingBox(
                            world_points=points, class_id=type_, class_description="SM"
                        )
                    )
        return sms

    def _get_visible_signs(self) -> List[BoundingBox]:

        signs = []
        for sec in self._get_visible_sections():
            for sign, height in self.get_traffic_signs_in_section(sec.id):
                if sign.intersects(self.camera_fov):
                    all_points = sign.get_points()
                    width = abs(Vector(all_points[0]) - Vector(all_points[-2]))
                    points = [
                        all_points[0] + Vector(0, 0, height),
                        all_points[0] + Vector(0, 0, height - width),
                    ] + list(
                        reversed(
                            [
                                all_points[-2] + Vector(0, 0, height - width),
                                all_points[-2] + Vector(0, 0, height),
                            ]
                        )
                    )
                    signs.append(
                        BoundingBox(
                            world_points=points, class_id=-2, class_description="sign"
                        )
                    )
        return signs

    def speak(
        self, image_size: Tuple[int, int], horizontal_fov: float
    ) -> List[BoundingBox]:
        """Create and return all bounding boxes of currently visible objects.

        Args:
            image_size: Total size of the image. Width and height.
            horizontal_fov: Field of view of the camera in horizontal direction.
        """

        # Get all bounding boxes sorted from nearest to furthest
        bbs = sorted(
            self._get_visible_obstacles()
            + self._get_visible_surface_markings()
            + self._get_visible_signs()
        )

        visible_bbs = []

        # Use this image to mark pixels that are already used by another bounding box
        # If a bounding is within another bounding box (and behind it),
        # it should not be returned.
        boxed_img = np.ones(image_size)

        for bb in bbs:
            # Filter bounding boxes that are not in the camera's field of view
            if abs(bb.angle) > math.pi / 2:
                continue
            bounds = bb.get_bounds()
            bounds = (
                max(bounds[0], 0),
                max(bounds[1], 0),
                min(bounds[2], image_size[0] - 1),
                min(bounds[3], image_size[1] - 1),
            )
            # Filter bounding boxes that are behind other boxes.
            if np.any(boxed_img[bounds[0] : bounds[2] + 1, bounds[1] : bounds[3] + 1]):
                visible_bbs.append(bb)
                boxed_img[bounds[0] : bounds[2] + 1, bounds[1] : bounds[3] + 1] = 0

        return visible_bbs
