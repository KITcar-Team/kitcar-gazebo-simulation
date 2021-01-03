import math
import os
import random
from typing import List

from simulation.utils.road.road import Road
from simulation.utils.road.sections import (
    LeftCircularArc,
    ParkingArea,
    ParkingLot,
    ParkingObstacle,
    ParkingSpot,
    RightCircularArc,
    StraightRoad,
)


def env_as_bool(name, default: bool):
    e = os.environ.get(name)
    if e is None:
        return default
    return e == "True" or e == "1"


NUM_PARKING_LOTS_RIGHT = int(os.environ.get("NUM_PARKING_LOTS_RIGHT", 1))
NUM_PARKING_LOTS_LEFT = int(os.environ.get("NUM_PARKING_LOTS_LEFT", 1))
CAN_PARK_RIGHT = env_as_bool("CAN_PARK_RIGHT", True)
CAN_PARK_LEFT = env_as_bool("CAN_PARK_LEFT", True)
CURVY_START = env_as_bool("CURVY_START", False)


road = Road()
road.append(StraightRoad(length=2))

if CURVY_START:
    road.append(
        random.choice([LeftCircularArc, RightCircularArc])(
            radius=random.random() * 2 + 1, angle=math.radians(30 + random.random() * 45)
        )
    )
    road.append(StraightRoad(length=1))


def get_random_spot(on_right_side: bool, kind: int, min_width: float = 0.3) -> ParkingSpot:
    width = (min_width + random.random()) if on_right_side else 0.35

    obstacle = None
    if kind == ParkingSpot.OCCUPIED or (
        kind == ParkingSpot.BLOCKED and random.random() > 0.5
    ):
        obs_width = (random.random() / 2 + 0.5) * width
        obstacle = ParkingObstacle(
            width=obs_width,
            _center=[0.2, -(width - (0.5 - random.random()) * obs_width / 2) / 2],
        )

    return ParkingSpot(width=width, kind=kind, obstacle=obstacle)


def get_spots(right_side: bool, can_park: bool) -> List[ParkingSpot]:
    available_kinds = [ParkingSpot.BLOCKED, ParkingSpot.OCCUPIED]
    if can_park:
        available_kinds.append(ParkingSpot.FREE)
    spots = [
        get_random_spot(right_side, kind=random.choice(available_kinds))
        for _ in range(random.randint(1, 4))
    ]

    if can_park:
        spots.insert(
            random.randint(0, len(spots)),
            get_random_spot(right_side, kind=ParkingSpot.FREE, min_width=0.7),
        )

    return spots


can_park = [False for _ in range(NUM_PARKING_LOTS_RIGHT)]
if CAN_PARK_RIGHT:
    can_park[random.randint(0, NUM_PARKING_LOTS_RIGHT - 1)] = True
right_lots = [
    ParkingLot(
        start=1 + i * 5,
        spots=get_spots(right_side=True, can_park=can_park[i]),
    )
    for i in range(NUM_PARKING_LOTS_RIGHT)
]


can_park = [False for _ in range(NUM_PARKING_LOTS_LEFT)]
if CAN_PARK_LEFT:
    can_park[random.randint(0, NUM_PARKING_LOTS_LEFT - 1)] = True
left_lots = [
    ParkingLot(
        start=1 + i * 5,
        spots=get_spots(right_side=False, can_park=can_park[i]),
    )
    for i in range(NUM_PARKING_LOTS_LEFT)
]


road.append(
    ParkingArea(
        length=5 * max(NUM_PARKING_LOTS_LEFT, NUM_PARKING_LOTS_RIGHT),
        start_line=True,
        left_lots=left_lots,
        right_lots=right_lots,
    )
)

road.append(StraightRoad(length=2))
