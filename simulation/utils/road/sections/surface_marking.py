from dataclasses import dataclass
from simulation.utils.road.sections.road_element import RoadElementRect, RoadElementPoly


@dataclass
class SurfaceMarking:
    """"""

    START_LINE = 0
    ZEBRA_CROSSING = 1
    BLOCKED_AREA = 2
    PARKING_SPOT_X = 3
    TRAFFIC_ISLAND_OPENING = 4
    TRAFFIC_ISLAND_CLOSING = 5

    RIGHT_TURN_MARKING = 10
    LEFT_TURN_MARKING = 11

    STOP_LINE = 21
    GIVE_WAY_LINE = 22

    kind: int = None
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
