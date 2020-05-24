"""Definition of the road.sections module

Collect classes and functions which should be included in the road.sections module.
"""
from road.sections.obstacle import StaticObstacle, ParkingObstacle  # noqa: 402
from road.sections.traffic_sign import TrafficSign  # noqa: 402
from road.sections.surface_marking import SurfaceMarking  # noqa: 402
from road.sections.surface_marking import SurfaceMarkingPoly  # noqa: 402
from road.sections.surface_marking import SurfaceMarkingRect  # noqa: 402
from road.sections.road_section import RoadSection  # noqa: 402
from road.sections.straight_road import StraightRoad  # noqa: 402
from road.sections.bezier_curve import QuadBezier, CubicBezier  # noqa: 402
from road.sections.circular_arc import LeftCircularArc, RightCircularArc  # noqa: 402
from road.sections.intersection import Intersection  # noqa: 402
from road.sections.parking_area import ParkingArea, ParkingLot, ParkingSpot  # noqa: 402
from road.sections.zebra_crossing import ZebraCrossing  # noqa: 402

__all__ = [
    "QuadBezier",
    "CubicBezier",
    "LeftCircularArc",
    "RightCircularArc",
    "Intersection",
    "StaticObstacle",
    "ParkingObstacle",
    "TrafficSign",
    "SurfaceMarkingPoly",
    "SurfaceMarkingRect",
    "ParkingArea",
    "ParkingLot",
    "ParkingSpot",
    "StartLine",
    "StraightRoad",
    "ZebraCrossing",
    "RoadSection",
]
