"""Left- and RightCircularArc."""

from dataclasses import dataclass
from typing import Tuple
import math

from simulation.utils.road.sections.road_section import RoadSection
import simulation.utils.road.sections.type as road_section_type
from simulation.utils.geometry import Point, Line, Pose


@dataclass
class _CircularArc(RoadSection):
    """Road section representing a part of a circle."""

    radius: float = None
    """Radius of the circle."""
    angle: float = None
    """Define the portion of the circle [radian]."""

    def __post_init__(self):
        assert self.radius is not None, "Missing argument radius for circular arc."
        assert self.angle is not None, "Missing argument angle for circular arc."

        super().__post_init__()

    @property
    def middle_line(self) -> Line:
        RADIAN_STEP = math.pi / 360
        points = []
        current_angle = 0
        radius = (
            -1 if self.__class__.TYPE == road_section_type.RIGHT_CIRCULAR_ARC else 1
        ) * self.radius
        while current_angle <= self.angle + RADIAN_STEP / 2:
            points.append(
                Point(
                    math.cos(current_angle - math.pi / 2) * abs(radius),
                    radius + math.sin(current_angle - math.pi / 2) * radius,
                )
            )

            current_angle += RADIAN_STEP

        return self.transform * Line(points)

    def get_ending(self) -> Tuple[Pose, float]:
        """Get the ending of the section as a pose and the curvature.

        Returns:
            A tuple consisting of the last point on the middle line together with \
            the direction facing along the middle line as a pose and the curvature \
            at the ending of the middle line.
        """
        _, curvature = super().get_ending()

        radius = (
            -1 if self.__class__.TYPE == road_section_type.RIGHT_CIRCULAR_ARC else 1
        ) * self.radius
        pose = self.transform * Pose(
            Point(
                math.cos(self.angle - math.pi / 2) * abs(radius),
                radius + math.sin(self.angle - math.pi / 2) * radius,
            ),
            -self.angle
            if self.__class__.TYPE == road_section_type.RIGHT_CIRCULAR_ARC
            else self.angle,
        )
        return (pose, curvature)


class LeftCircularArc(_CircularArc):
    """Part of a circle with a positive curvature.

    Args:
        radius (float): Radius [m].
        angle (float): Part of the circle [radian].
    """

    TYPE = road_section_type.LEFT_CIRCULAR_ARC


class RightCircularArc(_CircularArc):
    """Part of a circle with a negative curvature.

    Args:
        radius (float): Radius [m].
        angle (float): Part of the circle [radian].
    """

    TYPE = road_section_type.RIGHT_CIRCULAR_ARC
