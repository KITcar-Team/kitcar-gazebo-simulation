"""Road file for the examples in the onboarding documentation."""

import math

from simulation.utils.road.road import Road  # Definition of the road class
from simulation.utils.road.sections import (
    ParkingArea,
    ParkingLot,
    ParkingSpot,
    ParkingObstacle,
)
from simulation.utils.road.sections import Intersection
from simulation.utils.road.sections import ZebraCrossing
from simulation.utils.road.sections import LeftCircularArc
from simulation.utils.road.sections import StraightRoad
from simulation.utils.road.sections import StaticObstacle
from simulation.utils.geometry import Point

# - Beginning sphinx straight_road -
straight_road = StraightRoad(length=2)
# - Ending sphinx straight_road -

# - Beginning sphinx straight_road_obs -
straight_road_obs = StraightRoad(
    length=2, obstacles=[StaticObstacle(center=Point(1, 0), width=0.2, depth=0.1)]
)
# - Ending sphinx straight_road_obs -

# - Beginning sphinx parking_area -
parking_area = ParkingArea(
    length=4,
    start_line=True,
    left_lots=[
        ParkingLot(
            spots=[
                ParkingSpot(kind=ParkingSpot.OCCUPIED, obstacle=ParkingObstacle()),
                ParkingSpot(kind=ParkingSpot.BLOCKED),
            ],
        ),
        ParkingLot(
            start=2,
            opening_angle=math.radians(40),
            spots=[
                ParkingSpot(),
                ParkingSpot(kind=ParkingSpot.OCCUPIED, obstacle=ParkingObstacle()),
            ],
        ),
    ],
    right_lots=[
        ParkingLot(
            start=1,
            depth=0.4,
            spots=[
                ParkingSpot(kind=ParkingSpot.FREE, width=0.5),
                ParkingSpot(
                    kind=ParkingSpot.OCCUPIED, width=0.7, obstacle=ParkingObstacle()
                ),
                ParkingSpot(kind=ParkingSpot.BLOCKED),
            ],
        )
    ],
)
# - Ending sphinx parking_area -

# - Beginning sphinx intersection -
intersection = Intersection(size=2, turn=Intersection.RIGHT, angle=math.radians(110))
# - Ending sphinx intersection -

# - Beginning sphinx zebra_crossing -
zebra_crossing = ZebraCrossing(length=0.5)
# - Ending sphinx zebra_crossing -

# - Beginning sphinx left_arc -
left_arc = LeftCircularArc(radius=2, angle=math.radians(90))
# - Ending sphinx left_arc -

road = Road()
road.append(straight_road)
road.append(straight_road_obs)
road.append(parking_area)
road.append(intersection)
road.append(zebra_crossing)
road.append(left_arc)
