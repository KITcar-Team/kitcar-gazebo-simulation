"""Definition of the road.sections module

Collect classes and functions which should be included in the road.sections module.
"""
from road.sections.obstacle import StaticObstacle  # noqa: 402
from road.sections.straight_road import StraightRoad  # noqa: 402
from road.sections.road_section import RoadSection  # noqa: 402

__all__ = [
    "StaticObstacle",
    "StraightRoad",
    "RoadSection",
]
