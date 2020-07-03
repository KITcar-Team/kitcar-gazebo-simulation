import math
import random

from simulation.utils.road.road import Road  # Definition of the road class
from simulation.utils.road.sections import (
    StraightRoad,
    LeftCircularArc,
    RightCircularArc,
    StaticObstacle,
)


def angle():
    return math.radians(random.random() * 140 + 40)


def radius():
    return random.random() * 3 + 1


road = Road()
road.append(StraightRoad(length=4))
road.append(LeftCircularArc(radius=radius(), angle=angle(), obstacles=[StaticObstacle()]))
road.append(StraightRoad(length=2, obstacles=[StaticObstacle()]))
road.append(RightCircularArc(radius=radius(), angle=angle(), obstacles=[StaticObstacle()]))
road.append(StraightRoad(length=2))
