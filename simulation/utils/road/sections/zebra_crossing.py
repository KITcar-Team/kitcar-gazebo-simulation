""" ZebraCrossing."""

from dataclasses import dataclass

from geometry import Point, Polygon

from road.config import Config
from road.sections import StraightRoad
from road.sections.road_section import RoadSection
import road.sections.type as road_section_type


@dataclass
class ZebraCrossing(StraightRoad):
    TYPE = road_section_type.ZEBRA_CROSSING

    length: float = 0.45
    right_line_marking: int = RoadSection.SOLID_LINE_MARKING
    middle_line_marking: int = RoadSection.MISSING_LINE_MARKING
    left_line_marking: int = RoadSection.SOLID_LINE_MARKING

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

    def export(self):
        export = super().export()
        zebra = self.frame.to_schema_lanelet()
        zebra.type = "zebraCrossing"
        export.objects.append(zebra)
        return export
