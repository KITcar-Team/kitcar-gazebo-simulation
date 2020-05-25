from dataclasses import dataclass
from simulation.utils.road.sections.road_element import RoadElementRect

from typing import Tuple


@dataclass
class SignTuple:
    mesh: str
    collision_box_size: Tuple[float, float, float] = (0.11, 0.15, 0.29)
    collision_box_position: Tuple[float, float, float] = (-0.055, 0, 0.145)


@dataclass
class TrafficSign(RoadElementRect):
    ZONE_10_START = SignTuple("10zone_start")
    ZONE_20_START = SignTuple("20zone_start")
    ZONE_30_START = SignTuple("30zone_start")
    ZONE_40_START = SignTuple("40zone_start")
    ZONE_50_START = SignTuple("50zone_start")
    ZONE_60_START = SignTuple("60zone_start")
    ZONE_70_START = SignTuple("70zone_start")
    ZONE_80_START = SignTuple("80zone_start")
    ZONE_90_START = SignTuple("90zone_start")
    ZONE_10_END = SignTuple("10zone_end")
    ZONE_20_END = SignTuple("20zone_end")
    ZONE_40_END = SignTuple("40zone_end")
    ZONE_50_END = SignTuple("50zone_end")
    ZONE_60_END = SignTuple("60zone_end")
    ZONE_70_END = SignTuple("70zone_end")
    ZONE_80_END = SignTuple("80zone_end")
    ZONE_90_END = SignTuple("90zone_end")
    RAMP_END = SignTuple("Steigung_Ende")
    RAMP_START = SignTuple("Steigung_Start")
    YIELD = SignTuple("VorfahrtGewaehren")
    STOP = SignTuple("Stopschild")
    BARRED_AREA = SignTuple("Barred_area")
    TURN_LEFT = SignTuple("Links_abbiegen")
    TURN_RIGHT = SignTuple("Rechts_abbiegen")
    STVO_222 = SignTuple(
        "VorbeifahrtRechts_pedestrian", (0.11, 0.15, 0.15), (-0.055, 0, 0.075),
    )
    PRIORITY = SignTuple("Vorfahrt")
    STVO_350_10 = SignTuple("PedestrianCrossing",)
    CURVE_LEFT = SignTuple("Abbiegeschild_links", (0.09, 0.1, 0.11), (-0.045, 0, 0.055),)
    CURVE_LEFT_LARGE = SignTuple(
        "Abbiegeschild_gross_links", (0.09, 0.3, 0.11), (-0.045, 0, 0.055),
    )
    CURVE_RIGHT = SignTuple("Abbiegeschild_rechts", (0.09, 0.1, 0.11), (-0.045, 0, 0.055),)
    CURVE_RIGHT_LARGE = SignTuple(
        "Abbiegeschild_gross_links", (0.09, 0.3, 0.11), (-0.045, 0, 0.055),
    )

    kind: SignTuple = None

    def __post_init__(self):
        super().__post_init__()

        assert self.kind is not None, "Kind of the traffic sign not specified."
