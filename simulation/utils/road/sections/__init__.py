"""Definition of the road.sections module

Collect classes and functions which should be included in the road.sections module.
"""
from road.sections.obstacle import StaticObstacle, ParkingObstacle  # noqa: 402
from road.sections.straight_road import StraightRoad  # noqa: 402
from road.sections.bezier_curve import QuadBezier, CubicBezier  # noqa: 402
from road.sections.circular_arc import LeftCircularArc, RightCircularArc  # noqa: 402
from road.sections.parking_area import (
    ParkingArea,
    ParkingLot,
    ParkingSpot,
    StartLine,
)  # noqa: 402
from road.sections.road_section import RoadSection  # noqa: 402

__all__ = [
    "QuadBezier",
    "CubicBezier",
    "LeftCircularArc",
    "RightCircularArc",
    "StaticObstacle",
    "ParkingObstacle",
    "ParkingArea",
    "ParkingLot",
    "ParkingSpot",
    "StartLine",
    "StraightRoad",
    "RoadSection",
]
