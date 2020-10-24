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

    def _extract_bounding_boxes(self, func):
        lp_gen = (lp for sec in self._get_visible_sections() for lp in func(sec.id))
        bbs = []
        for lp in lp_gen:
            if lp.frame.intersects(self.camera_fov):
                points = lp.frame.get_points()
                points += [p + Vector(0, 0, lp.height) for p in points]
                bbs.append(
                    BoundingBox(
                        world_points=points, class_id=lp.id_, class_description=lp.desc
                    )
                )
        return bbs

    def _get_visible_obstacles(self) -> List[BoundingBox]:
        return self._extract_bounding_boxes(self.get_obstacles_in_section)

    def _get_visible_surface_markings(self) -> List[BoundingBox]:
        return self._extract_bounding_boxes(self.get_surface_markings_in_section)

    def _get_visible_signs(self) -> List[BoundingBox]:
        return self._extract_bounding_boxes(self.get_traffic_signs_in_section)

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
