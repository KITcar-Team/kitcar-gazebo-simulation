from dataclasses import dataclass
from typing import Tuple

from simulation.utils.road.sections.road_element import RoadElementRect

from . import ID


@dataclass
class SignTuple:
    mesh: str
    id_: int = None
    large_sign_collision_box_size = (0.015, 0.15, 0.3)
    large_sign_collision_box_position = (0, 0, 0.15)
    small_sign_collision_box_size = (0.015, 0.1, 0.275)
    small_sign_collision_box_position = (0, 0, 0.1375)
    collision_box_size: Tuple[float, float, float] = large_sign_collision_box_size
    collision_box_position: Tuple[float, float, float] = large_sign_collision_box_position

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

    PRIORITY = SignTuple(
        mesh="priority_sign",
        collision_box_size=SignTuple.small_sign_collision_box_size,
        collision_box_position=SignTuple.small_sign_collision_box_position,
    )
    YIELD = SignTuple(mesh="yield_sign")
    STOP = SignTuple(mesh="stop_sign")

    ONCOMING_TRAFFIC = SignTuple(
        mesh="oncoming_traffic_sign",
        collision_box_size=SignTuple.small_sign_collision_box_size,
        collision_box_position=SignTuple.small_sign_collision_box_position,
    )
    NO_OVERTAKING_START = SignTuple(
        mesh="no_overtaking_start_sign",
        collision_box_size=SignTuple.small_sign_collision_box_size,
        collision_box_position=SignTuple.small_sign_collision_box_position,
    )
    NO_OVERTAKING_END = SignTuple(
        mesh="no_overtaking_end_sign",
        collision_box_size=SignTuple.small_sign_collision_box_size,
        collision_box_position=SignTuple.small_sign_collision_box_position,
    )

    TURN_RIGHT = SignTuple(
        mesh="turn_right_sign",
        collision_box_size=SignTuple.small_sign_collision_box_size,
        collision_box_position=SignTuple.small_sign_collision_box_position,
    )
    TURN_LEFT = SignTuple(
        mesh="turn_left_sign",
        collision_box_size=SignTuple.small_sign_collision_box_size,
        collision_box_position=SignTuple.small_sign_collision_box_position,
    )
    PASS_RIGHT = SignTuple(
        mesh="pass_right_sign",
        collision_box_size=(0.015, 0.1, 0.125),
        collision_box_position=(0, 0, 0.0625),
    )
    PASS_LEFT = SignTuple(
        mesh="pass_left_sign",
        collision_box_size=(0.015, 0.1, 0.125),
        collision_box_position=(0, 0, 0.0625),
    )
    SHARP_TURN_RIGHT_SMALL = SignTuple(
        mesh="sharp_turn_right_small_sign",
        collision_box_size=(0.015, 0.1, 0.125),
        collision_box_position=(0, 0, 0.0625),
    )
    SHARP_TURN_RIGHT = SignTuple(
        mesh="sharp_turn_right_sign",
        collision_box_size=(0.015, 0.3, 0.125),
        collision_box_position=(0, 0, 0.0625),
    )
    SHARP_TURN_LEFT_SMALL = SignTuple(
        mesh="sharp_turn_left_small_sign",
        collision_box_size=(0.015, 0.1, 0.125),
        collision_box_position=(0, 0, 0.0625),
    )
    SHARP_TURN_LEFT = SignTuple(
        mesh="sharp_turn_left_sign",
        collision_box_size=(0.015, 0.3, 0.125),
        collision_box_position=(0, 0, 0.0625),
    )

    ZEBRA_CROSSING = SignTuple(mesh="zebra_crossing_sign")
    PARKING = SignTuple(mesh="parking_sign")
    EXPRESSWAY_START = SignTuple(mesh="expressway_start_sign")
    EXPRESSWAY_END = SignTuple(mesh="expressway_end_sign")

    def __init__(
        self,
        kind: SignTuple,
        arc_length: float,
        y: float = -0.5,
        angle=0,
        normalize_x: bool = True,
    ):

        self.kind = kind
        super().__init__(
            arc_length=arc_length,
            y=y,
            width=kind.collision_box_size[1],
            depth=kind.collision_box_size[0],
            angle=angle,
            normalize_x=normalize_x,
        )
