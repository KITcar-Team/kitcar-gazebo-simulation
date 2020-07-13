import math

from simulation.utils.road.road import Road  # Definition of the road class
from simulation.utils.road.sections import (
    StraightRoad,
    LeftCircularArc,
    RightCircularArc,
    StaticObstacle,
)

road = Road()
road.append(StraightRoad(length=4))
road.append(LeftCircularArc(radius=3, angle=math.radians(60), obstacles=[StaticObstacle()]))
road.append(StraightRoad(length=2, obstacles=[StaticObstacle()]))
road.append(
    RightCircularArc(radius=2, angle=math.radians(60), obstacles=[StaticObstacle()])
)
road.append(StraightRoad(length=2))
