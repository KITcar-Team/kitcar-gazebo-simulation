import os

from simulation.utils.road.road import Road
from simulation.utils.road.sections import (
    ParkingArea,
    ParkingLot,
    ParkingObstacle,
    ParkingSpot,
    StraightRoad,
)


def env_as_bool(name, default: bool):
    e = os.environ.get(name)
    if e is None:
        return default
    return e == "True" or e == "1"


PARKING_LOT_RIGHT = env_as_bool("PARKING_LOT_RIGHT", True)
PARKING_LOT_LEFT = env_as_bool("PARKING_LOT_LEFT", True)

road = Road()
road.append(StraightRoad(length=2))


right_lots = (
    [
        ParkingLot(
            start=1,
            spots=[
                ParkingSpot(
                    width=0.3, kind=ParkingSpot.OCCUPIED, obstacle=ParkingObstacle()
                ),
                ParkingSpot(width=0.4, kind=ParkingSpot.BLOCKED),
                ParkingSpot(width=1),
                ParkingSpot(
                    width=0.3, kind=ParkingSpot.OCCUPIED, obstacle=ParkingObstacle()
                ),
            ],
        )
    ]
    if PARKING_LOT_RIGHT
    else []
)


left_lots = (
    [
        ParkingLot(
            start=1,
            spots=[
                ParkingSpot(kind=ParkingSpot.OCCUPIED, obstacle=ParkingObstacle()),
                ParkingSpot(kind=ParkingSpot.BLOCKED),
                ParkingSpot(),
                ParkingSpot(kind=ParkingSpot.OCCUPIED, obstacle=ParkingObstacle()),
            ],
        )
    ]
    if PARKING_LOT_LEFT
    else []
)

road.append(
    ParkingArea(
        length=5,
        start_line=True,
        left_lots=left_lots,
        right_lots=right_lots,
    )
)

road.append(StraightRoad(length=2))
