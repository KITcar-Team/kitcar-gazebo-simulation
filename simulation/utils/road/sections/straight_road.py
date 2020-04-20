"""The StraightRoad can be used to create straight sections of a road.

As any other road sections, line markings can be variied and obstacles created on the road.
"""

from dataclasses import dataclass

from geometry import Point, Line

from road.sections.road_section import RoadSection
import road.sections.type as road_section_type


@dataclass
class _StraightRoad(RoadSection):

    TYPE = road_section_type.STRAIGHT_ROAD

    length: float = 1
    """Length of the section."""

    def __post_init__(self):
        assert self.length > 0, "Invalid: length for StraightRoad is smaller than 0."
        super().__post_init__()


class StraightRoad(_StraightRoad):
    """Straight section of the road.

    Example:
        >>> from road.sections import StraightRoad
        >>> from road.road import Road
        >>> road = Road()
        >>> road.append(StraightRoad(length=2))
        >>> road
        Road(seed='KITCAR', sections=[ID=0, StraightRoad(length=2)])
    """

    @property
    def middle_line(self) -> Line:
        return self.transform * Line([Point(0, 0), Point(self.length, 0)])
