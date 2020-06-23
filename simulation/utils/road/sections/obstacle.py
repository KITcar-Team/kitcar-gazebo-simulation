"""StaticObstacle on road and ParkingObstacle on ParkingSpot."""
from dataclasses import dataclass

from simulation.utils.road.sections.road_element import RoadElementRect
from simulation.utils.geometry import Point


@dataclass
class StaticObstacle(RoadElementRect):
    height: float = 0.2
    """Height of the obstacle."""


@dataclass
class ParkingObstacle(StaticObstacle):
    center: Point = Point(0.2, -0.2)
    """Center point of the obstacle."""
    width: float = 0.15
    """Width of the obstacle."""
    depth: float = 0.15
    """Width of the obstacle."""
    normalize_x: bool = False
