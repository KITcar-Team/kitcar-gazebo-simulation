"""The default road when launching this simulation."""
import math

from simulation.utils.road.road import Road  # Definition of the road class
from simulation.utils.road.sections import (
    StraightRoad,
    LeftCircularArc,
    ZebraCrossing,
    Intersection,
    ParkingArea,
    ParkingLot,
    ParkingSpot,
    ParkingObstacle,
    StaticObstacle,
)
from simulation.utils.road.sections.road_section import RoadSection

road = Road()
road.append(StraightRoad(length=1))

# Create a parking area with different kinds of parking spots
road.append(
    ParkingArea(
        length=4,
        start_line=True,
        left_lots=[
            ParkingLot(
                start=1, spots=[ParkingSpot(), ParkingSpot(kind=ParkingSpot.BLOCKED)]
            )
        ],
        right_lots=[
            ParkingLot(
                start=0.2,
                spots=[
                    ParkingSpot(
                        kind=ParkingSpot.OCCUPIED,
                        width=0.7,
                        obstacle=ParkingObstacle(center=[0.2, -0.2], width=0.3, depth=0.4),
                    ),
                    ParkingSpot(kind=ParkingSpot.BLOCKED),
                ],
            )
        ],
    )
)
road.append(
    LeftCircularArc(
        radius=2, angle=math.pi / 2, right_line_marking=RoadSection.MISSING_LINE_MARKING
    )
)
road.append(StraightRoad(length=0.45))
road.append(LeftCircularArc(radius=2, angle=math.pi / 2))
road.append(Intersection(size=3, turn=Intersection.RIGHT))
road.append(LeftCircularArc(radius=1.5, angle=math.pi, obstacles=[StaticObstacle()]))
road.append(Intersection(size=3, turn=Intersection.RIGHT))
road.append(
    LeftCircularArc(
        radius=1.5, angle=math.pi / 2, left_line_marking=RoadSection.MISSING_LINE_MARKING
    )
)
road.append(ZebraCrossing())
road.append(StraightRoad(length=1))
road.append(LeftCircularArc(radius=1.5, angle=math.pi / 2))
road.append(StraightRoad(length=1))
