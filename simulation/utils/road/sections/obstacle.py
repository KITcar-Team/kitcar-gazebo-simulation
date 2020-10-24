"""StaticObstacle on road and ParkingObstacle on ParkingSpot."""
from dataclasses import dataclass

from simulation.utils.geometry import Point, Transform, Vector
from simulation.utils.road.sections.road_element import RoadElementRect

from . import ID


@dataclass
class StaticObstacle(RoadElementRect):
    id_ = ID.register()
    desc = "StaticObstacle"
    height: float = 0.2
    """Height of the obstacle."""


@dataclass
class _ParkingObstacle(StaticObstacle):
    center: Point = Point(0.2, -0.2)
    """Center point of the obstacle."""


@dataclass
class ParkingObstacle(_ParkingObstacle):
    id_ = ID.register()
    desc = "ParkingObstacle"

    width: float = 0.15
    """Width of the obstacle."""
    depth: float = 0.15
    """Width of the obstacle."""
    normalize_x: bool = False

    @property
    def center(self) -> Point:
        """Point: Center of the element in global coordinates."""
        tf = Transform([-self._center.x, 0], 0) if self.normalize_x else 1
        return Point(self.transform * (tf * Vector(self._center)))

    @center.setter
    def center(self, c: Point):
        if not type(c) is Point:
            c = Point(c)
        self._center = c
