"""Left- and RightCircularArc."""

from dataclasses import dataclass
import math

from road.sections.road_section import RoadSection
import road.sections.type as road_section_type
from geometry import Point, Line


@dataclass
class _CircularArc(RoadSection):
    """Road section representing a part of a circle."""

    radius: float = None
    """Radius of the circle."""
    angle: float = None
    """Define the portion of the circle [degree]."""

    def __post_init__(self):
        assert self.radius is not None, "Missing argument radius for circular arc."
        assert self.angle is not None, "Missing argument angle for circular arc."

        super().__post_init__()

    @property
    def radians(self) -> float:
        """float: Angle converted into radians."""
        return math.radians(self.angle)

    @property
    def middle_line(self) -> Line:
        RADIAN_STEP = math.pi / 360
        points = []
        current_angle = 0
        radius = (
            -1 if self.__class__.TYPE == road_section_type.RIGHT_CIRCULAR_ARC else 1
        ) * self.radius
        while current_angle <= self.radians + RADIAN_STEP / 2:
            points.append(
                Point(
                    math.cos(current_angle - math.pi / 2) * abs(radius),
                    radius + math.sin(current_angle - math.pi / 2) * radius,
                )
            )

            current_angle += RADIAN_STEP

        return self.transform * Line(points)


class LeftCircularArc(_CircularArc):
    """Part of a circle with a positive curvature."""

    TYPE = road_section_type.LEFT_CIRCULAR_ARC


class RightCircularArc(_CircularArc):
    """Part of a circle with a negative curvature."""

    TYPE = road_section_type.RIGHT_CIRCULAR_ARC
