from dataclasses import dataclass
from typing import Tuple

from simulation.utils.road.sections.road_element import RoadElementPoly, RoadElementRect

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

    kind: Tuple[int, str] = None
    """Kind of the surface marking."""

    def __post_init__(self):
        super().__post_init__()

        assert self.kind is not None, "Kind of SurfaceMarking not specified."


@dataclass
class SurfaceMarkingPoly(RoadElementPoly, SurfaceMarking):
    pass


@dataclass
class SurfaceMarkingRect(RoadElementRect, SurfaceMarking):
    pass
