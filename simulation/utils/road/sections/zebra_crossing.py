""" ZebraCrossing."""

from dataclasses import dataclass

from simulation.utils.geometry import Point, Polygon

from simulation.utils.road.config import Config
from simulation.utils.road.sections import StraightRoad, SurfaceMarkingRect
import simulation.utils.road.sections.type as road_section_type


@dataclass
class ZebraCrossing(StraightRoad):
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
        poly = Polygon(
            [
                Point(0, -Config.road_width),
                Point(0, Config.road_width),
                Point(self.length, Config.road_width),
                Point(self.length, -Config.road_width),
            ]
        )
        return self.transform * poly
