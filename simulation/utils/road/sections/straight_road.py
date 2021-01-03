"""The StraightRoad can be used to create straight sections of a road.

As any other road sections, line markings can be variied and obstacles created on the road.
"""

from dataclasses import dataclass

import simulation.utils.road.sections.type as road_section_type
from simulation.utils.geometry import Line, Point
from simulation.utils.road.sections.road_section import RoadSection


@dataclass
class StraightRoad(RoadSection):
    """Straight section of the road.

    Args:
        length (float): Length [m] of the section.

    Example:
        >>> from simulation.utils.road.sections import StraightRoad
        >>> from simulation.utils.road.road import Road
        >>> road = Road()
        >>> road.append(StraightRoad(length=2))
        >>> road
        Road(_name=None, _seed=None, use_seed=True, \
sections=[StraightRoad(id=0, \
transform=Transform(translation=Vector(0.0, 0.0, 0.0),\
rotation=Quaternion(1.0, 0.0, 0.0, 0.0)), is_start=False, left_line_marking='solid', \
middle_line_marking='dashed', right_line_marking='solid', obstacles=[], traffic_signs=[], \
surface_markings=[], length=2)], length=2.0)
    """

    TYPE = road_section_type.STRAIGHT_ROAD

    length: float = 1
    """Length of the section."""

    def __post_init__(self):
        assert self.length > 0, "Invalid: length for StraightRoad is smaller than 0."
        super().__post_init__()

    @property
    def middle_line(self) -> Line:
        return self.transform * Line([Point(0, 0), Point(self.length, 0)])
