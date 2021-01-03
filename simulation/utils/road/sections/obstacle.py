"""StaticObstacle on road and ParkingObstacle on ParkingSpot."""
from dataclasses import dataclass

from simulation.utils.geometry import Point
from simulation.utils.road.sections.road_element import RoadElementRect

from . import ID


@dataclass
class StaticObstacle(RoadElementRect):
    id_ = ID.register()
    desc = "StaticObstacle"
    height: float = 0.2
    """Height of the obstacle."""


@dataclass
class ParkingObstacle(StaticObstacle):
    id_ = ID.register()
    desc = "ParkingObstacle"

    _center: Point = Point(0.2, -0.2)
    """Center point of the obstacle."""
    width: float = 0.15
    """Width of the obstacle."""
    depth: float = 0.15
    """Width of the obstacle."""
    normalize_x: bool = False
