# MRT Maschinenhalle complete road
# as of 2020-02
import math

from simulation.utils.road.road import Road
from simulation.utils.road.sections import (
    Intersection,
    LeftCircularArc,
    ParkingArea,
    ParkingLot,
    ParkingObstacle,
    ParkingSpot,
    RightCircularArc,
    StraightRoad,
)

road = Road()
road.append(ParkingArea(length=0.06, start_line=True))
road.append(StraightRoad(length=0.62))
road.append(
    ParkingArea(
        length=2.9,
        right_lots=[
            ParkingLot(
                depth=0.32,
                spots=[
                    ParkingSpot(width=1.15),
                    ParkingSpot(
                        width=0.25, kind=ParkingSpot.OCCUPIED, obstacle=ParkingObstacle()
                    ),
                    ParkingSpot(width=0.045),
                    ParkingSpot(width=0.56, kind=ParkingSpot.BLOCKED),
                    ParkingSpot(width=0.32),
                ],
            )
        ],
    )
)
road.append(StraightRoad(length=0.53))
road.append(
    ParkingArea(
        length=3.8,
        left_lots=[
            ParkingLot(
                opening_angle=math.radians(50),
                depth=0.52,
                spots=[
                    ParkingSpot(width=0.37),
                    ParkingSpot(width=0.37, kind=ParkingSpot.BLOCKED),
                    ParkingSpot(width=0.35),
                    ParkingSpot(width=0.35),
                    ParkingSpot(width=0.35),
                ],
            )
        ],
        right_lots=[
            ParkingLot(
                opening_angle=math.radians(48),
                depth=0.32,
                spots=[
                    ParkingSpot(width=0.2),
                    ParkingSpot(
                        width=0.11, kind=ParkingSpot.OCCUPIED, obstacle=ParkingObstacle()
                    ),
                    ParkingSpot(width=0.71),
                    ParkingSpot(
                        width=0.11, kind=ParkingSpot.OCCUPIED, obstacle=ParkingObstacle()
                    ),
                    ParkingSpot(width=0.56),
                    ParkingSpot(width=0.73, kind=ParkingSpot.BLOCKED),
                    ParkingSpot(width=1.14),
                ],
            )
        ],
    )
)
road.append(LeftCircularArc(radius=1.41, angle=math.radians(140)))
road.append(StraightRoad(length=1.53))
road.append(Intersection(angle=math.radians(70), size=1.8))
road.append(RightCircularArc(radius=1.285, angle=math.radians(290)))
road.append(Intersection(angle=math.radians(110), size=1.8))
road.append(RightCircularArc(radius=1.81, angle=math.radians(50)))
road.append(LeftCircularArc(radius=2.24, angle=math.radians(20)))
road.append(StraightRoad(length=3.94))
road.append(LeftCircularArc(radius=1.26, angle=math.radians(90)))
road.append(StraightRoad(length=1.06))
road.append(LeftCircularArc(radius=1.28, angle=math.radians(90)))
road.append(StraightRoad(length=0.05))
