"""A simple road for the onboarding task."""
import math

from simulation.utils.road.road import Road  # Definition of the road class
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
road.append(StraightRoad())
parking_area = ParkingArea(
    length=3.5,
    start_line=True,
    left_lots=[
        ParkingLot(
            start=0.5,
            depth=0.5,
            spots=[
                ParkingSpot(kind=ParkingSpot.BLOCKED),
                ParkingSpot(),
                ParkingSpot(kind=ParkingSpot.OCCUPIED, obstacle=ParkingObstacle()),
            ],
        ),
    ],
    right_lots=[
        ParkingLot(
            start=1.5,
            spots=[
                ParkingSpot(width=0.7, kind=ParkingSpot.BLOCKED),
                ParkingSpot(width=0.7),
            ],
        )
    ],
)
road.append(parking_area)
road.append(LeftCircularArc(radius=2, angle=math.radians(90)))
road.append(LeftCircularArc(radius=1.5, angle=math.radians(90)))
road.append(RightCircularArc(radius=1.5, angle=math.radians(90)))
road.append(LeftCircularArc(radius=2.5, angle=math.radians(60)))
road.append(StraightRoad(length=0.8))
road.append(LeftCircularArc(radius=2.5, angle=math.radians(120)))
road.append(Intersection(size=3.2, angle=math.radians(110)))
road.append(StraightRoad(length=1.66))
road.append(LeftCircularArc(radius=1, angle=math.radians(90)))
road.close_loop()
