from dataclasses import dataclass
from typing import Tuple

from simulation.utils.road.sections.road_element import RoadElement, RoadElementRect

from . import ID

ID_NS = 200


@dataclass
class SurfaceMarking:
    """"""

    START_LINE = (ID.register(ID_NS), "Startline")
    ZEBRA_CROSSING = (ID.register(ID_NS), "CrosswalkLines")
    BLOCKED_AREA = (ID.register(ID_NS), "Blockedarea")
    PARKING_SPOT_X = (ID.register(ID_NS), "ParkingspotX")
    TRAFFIC_ISLAND_BLOCKED = (ID.register(ID_NS), "Trafficisland")
    ZEBRA_LINES = (ID.register(ID_NS), "CrosswalkNoLines")

    RIGHT_TURN_MARKING = (ID.register(ID_NS), "RightArrow")
    LEFT_TURN_MARKING = (ID.register(ID_NS), "LeftArrow")

    STOP_LINE = (ID.register(ID_NS), "Stopline")
    GIVE_WAY_LINE = (ID.register(ID_NS), "GiveWayLine")

    ZONE_10_START = (ID.register(ID_NS), "ZONE_10_START")
    ZONE_20_START = (ID.register(ID_NS), "ZONE_20_START")
    ZONE_30_START = (ID.register(ID_NS), "ZONE_30_START")
    ZONE_40_START = (ID.register(ID_NS), "ZONE_40_START")
    ZONE_50_START = (ID.register(ID_NS), "ZONE_50_START")
    ZONE_60_START = (ID.register(ID_NS), "ZONE_60_START")
    ZONE_70_START = (ID.register(ID_NS), "ZONE_70_START")
    ZONE_80_START = (ID.register(ID_NS), "ZONE_80_START")
    ZONE_90_START = (ID.register(ID_NS), "ZONE_90_START")
    ZONE_10_END = (ID.register(ID_NS), "ZONE_10_END")
    ZONE_20_END = (ID.register(ID_NS), "ZONE_20_END")
    ZONE_30_END = (ID.register(ID_NS), "ZONE_30_END")
    ZONE_40_END = (ID.register(ID_NS), "ZONE_40_END")
    ZONE_50_END = (ID.register(ID_NS), "ZONE_50_END")
    ZONE_60_END = (ID.register(ID_NS), "ZONE_60_END")
    ZONE_70_END = (ID.register(ID_NS), "ZONE_70_END")
    ZONE_80_END = (ID.register(ID_NS), "ZONE_80_END")
    ZONE_90_END = (ID.register(ID_NS), "ZONE_90_END")

    kind: Tuple[int, str] = None
    """Kind of the surface marking."""

    def __post_init__(self):
        super().__post_init__()

        assert self.kind is not None, "Kind of SurfaceMarking not specified."


@dataclass
class SurfaceMarkingPoly(RoadElement, SurfaceMarking):
    pass


@dataclass
class SurfaceMarkingRect(RoadElementRect, SurfaceMarking):
    pass
