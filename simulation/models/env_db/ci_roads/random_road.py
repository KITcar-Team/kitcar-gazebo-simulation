import math
import random

from simulation.utils.geometry import Point
from simulation.utils.road.road import Road
from simulation.utils.road.sections import (
    BlockedArea,
    Intersection,
    LeftCircularArc,
    ParkingArea,
    ParkingLot,
    ParkingObstacle,
    ParkingSpot,
    RightCircularArc,
    StaticObstacle,
    StraightRoad,
    TrafficIsland,
    ZebraCrossing,
)

road = Road()
road.append(StraightRoad(length=1))

ROAD_WIDTH = 0.4


def get_random_parking() -> ParkingArea:
    """Create a randomized parking area."""

    def get_spot(side=0):
        width = (0.5 + random.random()) * (0.2 + ROAD_WIDTH * side)
        obs_width = (random.random() / 2 + 0.5) * width
        return ParkingSpot(
            width=width,
            kind=random.choice([ParkingSpot.FREE, ParkingSpot.BLOCKED]),
            obstacle=random.choice(
                [
                    ParkingObstacle(
                        width=obs_width,
                        _center=[0.2, -(width - random.random() / 2 * obs_width) / 2],
                    ),
                    ParkingObstacle(
                        width=obs_width,
                        _center=[0.2, -(width - random.random() / 2 * obs_width) / 2],
                    ),
                    None,
                ]
            ),
        )

    return ParkingArea(
        length=7,
        start_line=True,
        left_lots=[
            ParkingLot(
                start=random.random() * 2,
                spots=[get_spot(0) for _ in range(random.randint(1, 5))],
            )
        ],
        right_lots=[
            ParkingLot(
                start=random.random() * 2,
                spots=[get_spot(1) for _ in range(random.randint(1, 5))],
            )
        ],
    )


def get_random_obstacle(
    x_range=(0, 1),
    y_range=(-ROAD_WIDTH, ROAD_WIDTH),
    width_range=(0.1, 0.3),
    depth_range=(0.1, 0.3),
    height_range=(0.1, 0.4),
) -> StaticObstacle:
    """Create a randomized obstacle."""

    def draw_from(r):
        return (r[1] - r[0]) * random.random() + r[0]

    return StaticObstacle(
        _center=Point(draw_from(x_range), draw_from(y_range)),
        width=draw_from(width_range),
        height=draw_from(height_range),
        depth=draw_from(depth_range),
    )


for _ in range(20):
    road.append(
        random.choices(
            [
                get_random_parking(),
                LeftCircularArc(
                    radius=3 * random.random() + 0.5,
                    angle=0.1 + math.pi / 2 * random.random(),
                ),
                RightCircularArc(
                    radius=3 * random.random() + 0.5,
                    angle=0.1 + math.pi / 2 * random.random(),
                ),
                Intersection(
                    size=2,
                    turn=random.choices(
                        [Intersection.RIGHT, Intersection.LEFT, Intersection.STRAIGHT],
                        weights=[1, 1, 2],
                        k=1,
                    )[0],
                    angle=(70 + 40 * random.random()) * math.pi / 180,
                    rule=random.choices(
                        [
                            Intersection.EQUAL,
                            Intersection.PRIORITY_STOP,
                            Intersection.PRIORITY_YIELD,
                            Intersection.STOP,
                            Intersection.YIELD,
                        ],
                        weights=[1, 1, 1, 1, 1],
                        k=1,
                    )[0],
                ),
                BlockedArea(obstacles=random.choice([[get_random_obstacle()], []])),
                ZebraCrossing(),
                StraightRoad(
                    obstacles=random.choice([[get_random_obstacle()], []]),
                ),
                TrafficIsland(
                    zebra_marking_type=random.choice(
                        [TrafficIsland.LINES, TrafficIsland.ZEBRA]
                    )
                ),
            ],
            weights=[1, 4, 4, 2, 2, 2, 4, 1],
            k=1,
        )[0]
    )
road.close_loop()
