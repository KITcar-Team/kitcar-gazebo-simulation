from dataclasses import dataclass
from typing import Tuple

from simulation.utils.road.sections.road_element import RoadElementRect

from . import ID


@dataclass
class SignTuple:
    mesh: str
    id_: int = None
    collision_box_size: Tuple[float, float, float] = (0.11, 0.15, 0.29)
    collision_box_position: Tuple[float, float, float] = (-0.055, 0, 0.145)

    def __post_init__(self):
        self.id_ = ID.register(ns=100)


@dataclass
class TrafficSign(RoadElementRect):
    ZONE_10_START = SignTuple(mesh="speed_limit_zone_10_start_sign")
    ZONE_20_START = SignTuple(mesh="speed_limit_zone_20_start_sign")
    ZONE_30_START = SignTuple(mesh="speed_limit_zone_30_start_sign")
    ZONE_40_START = SignTuple(mesh="speed_limit_zone_40_start_sign")
    ZONE_50_START = SignTuple(mesh="speed_limit_zone_50_start_sign")
    ZONE_60_START = SignTuple(mesh="speed_limit_zone_60_start_sign")
    ZONE_70_START = SignTuple(mesh="speed_limit_zone_70_start_sign")
    ZONE_80_START = SignTuple(mesh="speed_limit_zone_80_start_sign")
    ZONE_90_START = SignTuple(mesh="speed_limit_zone_90_start_sign")
    ZONE_10_END = SignTuple(mesh="speed_limit_zone_10_end_sign")
    ZONE_20_END = SignTuple(mesh="speed_limit_zone_20_end_sign")
    ZONE_30_END = SignTuple(mesh="speed_limit_zone_30_end_sign")
    ZONE_40_END = SignTuple(mesh="speed_limit_zone_40_end_sign")
    ZONE_50_END = SignTuple(mesh="speed_limit_zone_50_end_sign")
    ZONE_60_END = SignTuple(mesh="speed_limit_zone_60_end_sign")
    ZONE_70_END = SignTuple(mesh="speed_limit_zone_70_end_sign")
    ZONE_80_END = SignTuple(mesh="speed_limit_zone_80_end_sign")
    ZONE_90_END = SignTuple(mesh="speed_limit_zone_90_end_sign")

    RAMP_START = SignTuple(mesh="uphill_grade_sign")
    RAMP_END = SignTuple(mesh="downhill_grade_sign")

    PRIORITY = SignTuple(mesh="priority_sign")
    YIELD = SignTuple(mesh="yield_sign")
    STOP = SignTuple(mesh="stop_sign")

    ONCOMING_TRAFFIC = SignTuple(mesh="oncoming_traffic_sign")
    NO_OVERTAKING_START = SignTuple(mesh="no_overtaking_start_sign")
    NO_OVERTAKING_END = SignTuple(mesh="no_overtaking_end_sign")

    TURN_RIGHT = SignTuple(mesh="turn_right_sign")
    TURN_LEFT = SignTuple(mesh="turn_left_sign")
    PASS_RIGHT = SignTuple(mesh="pass_right_sign")
    PASS_LEFT = SignTuple(mesh="pass_left_sign")
    SHARP_TURN_RIGHT_SMALL = SignTuple(mesh="sharp_turn_right_small_sign")
    SHARP_TURN_RIGHT = SignTuple(mesh="sharp_turn_right_sign")
    SHARP_TURN_LEFT_SMALL = SignTuple(mesh="sharp_turn_left_small_sign")
    SHARP_TURN_LEFT = SignTuple(mesh="sharp_turn_left_sign")

    ZEBRA_CROSSING = SignTuple(mesh="zebra_crossing_sign")
    PARKING = SignTuple(mesh="parking_sign")
    EXPRESSWAY_START = SignTuple(mesh="expressway_start_sign")
    EXPRESSWAY_END = SignTuple(mesh="expressway_end_sign")

    kind: SignTuple = None

    def __post_init__(self):
        super().__post_init__()

        assert self.kind is not None, "Kind of the traffic sign not specified."
