"""StaticObstacle on road and ParkingObstacle on ParkingSpot."""
from dataclasses import dataclass

from simulation.utils.road.sections.road_element import RoadElementRect

from . import ID


@dataclass
class StaticObstacle(RoadElementRect):
    """Obstacle that can be placed on the road.

    Args:
        arc_length: x coordinate of the element along the road.
        y: y coordinate of the element. (Perpendicular to the road.)
        width: Width of the element.
        depth: Depth of the element. Component of the size in the direction of the road.
        height: Height of the element.
        angle: Angle [radian] between the middle line and the element
            (measured at the center).
    """

    id_ = ID.register()
    desc = "StaticObstacle"

    def __init__(
        self,
        arc_length: float = 0.4,
        y: float = -0.2,
        width: float = 0.2,
        depth: float = 0.2,
        height: float = 0.2,
        angle=0,
    ):
        super().__init__(arc_length, y, width, depth, angle)
        self.height = height


@dataclass
class ParkingObstacle(StaticObstacle):
    """Obstacle that can be placed on a parking spot.

    Args:
        x: x coordinate of the element along the road.
        y: y coordinate of the element. (Perpendicular to the road.)
        width: Width of the element.
        depth: Depth of the element. Component of the size in the direction of the spot.
        height: Height of the element.
        angle: Angle [radian] between the parking spot and the element
            (measured at the center).
    """

    id_ = ID.register()
    desc = "ParkingObstacle"

    def __init__(
        self,
        x: float = 0.15,
        y: float = -0.15,
        width: float = 0.15,
        depth: float = 0.15,
        height: float = 0.2,
        angle=0,
    ):
        super().__init__(x, y, width, depth, height, angle)
        self.normalize_x = False
