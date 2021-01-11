"""Road file for the examples in the onboarding documentation."""

import math

from simulation.utils.road.road import Road  # Definition of the road class
from simulation.utils.road.sections import (
    BlockedArea,
    Intersection,
    LeftCircularArc,
    ParkingArea,
    ParkingLot,
    ParkingObstacle,
    ParkingSpot,
    StraightRoad,
    TrafficIsland,
    ZebraCrossing,
)

# - Beginning sphinx straight_road -
straight_road = StraightRoad(length=2)
# - Ending sphinx straight_road -

# - Beginning sphinx straight_road_obs -
straight_road_obs = StraightRoad(length=2)
straight_road_obs.add_obstacle(arc_length=1, y_offset=0, width=0.2, length=0.1)
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

# - Beginning sphinx blocked_area -
blocked_area = BlockedArea(length=1, width=0.2)
# - Ending sphinx blocked_area -

# - Beginning sphinx traffic_island -
traffic_island = TrafficIsland(
    island_width=0.3,
    zebra_length=0.45,
    curve_area_length=0.8,
    curvature=0.4,
    zebra_marking_type=TrafficIsland.ZEBRA,
)
# - Ending sphinx traffic_island -

road = Road()
road.append(straight_road)
road.append(straight_road_obs)
road.append(parking_area)
road.append(intersection)
road.append(zebra_crossing)
road.append(left_arc)
road.append(blocked_area)
road.append(traffic_island)
