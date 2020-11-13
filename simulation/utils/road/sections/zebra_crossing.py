"""ZebraCrossing."""

from dataclasses import dataclass
from typing import List

import simulation.utils.road.sections.type as road_section_type
from simulation.utils.geometry import Point, Polygon
from simulation.utils.road.config import Config
from simulation.utils.road.sections import StraightRoad, SurfaceMarkingRect, TrafficSign


@dataclass
class ZebraCrossing(StraightRoad):
    """Road section representing a zebra crossing.

    Args:
        length (float) = 0.45: length of the crossing and thus the section.
    """

    TYPE = road_section_type.ZEBRA_CROSSING

    length: float = 0.45

    def __post_init__(self):
        self.surface_markings.append(
            SurfaceMarkingRect(
                center=Point(self.length / 2, 0),
                depth=self.length,
                width=2 * Config.road_width,
                kind=SurfaceMarkingRect.ZEBRA_CROSSING,
            )
        )
        self.middle_line_marking = self.MISSING_LINE_MARKING
        super().__post_init__()

    @property
    def frame(self) -> Polygon:
        """Polygon : Frame for the zebra crossing surface marking."""
        poly = Polygon(
            [
                Point(0, -Config.road_width),
                Point(0, Config.road_width),
                Point(self.length, Config.road_width),
                Point(self.length, -Config.road_width),
            ]
        )
        return self.transform * poly

    @property
    def traffic_signs(self) -> List[TrafficSign]:
        """List[TrafficSign]: All traffic signs within this section of the road."""
        traffic_signs = super().traffic_signs.copy()

        traffic_signs.append(
            TrafficSign(
                kind=TrafficSign.ZEBRA_CROSSING,
                center=self.transform * Point(-0.4, -Config.road_width - 0.1),
                angle=self.transform.get_angle(),
                normalize_x=False,
            )
        )
        return traffic_signs

    @traffic_signs.setter
    def traffic_signs(self, signs: List[TrafficSign]):
        self._traffic_signs = signs
