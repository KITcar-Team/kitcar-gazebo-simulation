"""BlockedArea."""

import math
from dataclasses import dataclass
from typing import List

import simulation.utils.road.sections.type as road_section_type
from simulation.utils.geometry import Point, Polygon
from simulation.utils.road.config import Config
from simulation.utils.road.sections import StraightRoad, SurfaceMarkingPoly, TrafficSign


@dataclass
class BlockedArea(StraightRoad):
    """Road section representing a blocked area.

    Args:
        width (float) = 0.2: width of the blocked area, starting from the right line.
    """

    TYPE = road_section_type.BLOCKED_AREA

    width: float = 0.2
    _opening_angle: float = math.radians(60)

    def __post_init__(self):
        self.surface_markings.append(
            SurfaceMarkingPoly(
                frame=self._poly, kind=SurfaceMarkingPoly.BLOCKED_AREA, normalize_x=False
            )
        )
        super().__post_init__()

    @property
    def frame(self) -> Polygon:
        """Polygon: Frame of the blocked area surface marking.

        It has the shape of a symmetrical trapezoid.
        """
        return self.transform * self._poly

    @property
    def _poly(self) -> Polygon:
        opening_x = self.width / math.tan(self._opening_angle)
        return Polygon(
            [
                Point(0, -Config.road_width),
                Point(opening_x, -Config.road_width + self.width),
                Point(self.length - opening_x, -Config.road_width + self.width),
                Point(self.length, -Config.road_width),
            ]
        )

    @property
    def traffic_signs(self) -> List[TrafficSign]:
        """List[TrafficSign]: All traffic signs within this section of the road."""
        traffic_signs = super().traffic_signs.copy()

        traffic_signs.append(
            TrafficSign(
                kind=TrafficSign.ONCOMING_TRAFFIC,
                center=self.transform * Point(-0.4, -Config.road_width - 0.1),
                angle=self.transform.get_angle(),
                normalize_x=False,
            )
        )
        return traffic_signs

    @traffic_signs.setter
    def traffic_signs(self, signs: List[TrafficSign]):
        self._traffic_signs = signs
