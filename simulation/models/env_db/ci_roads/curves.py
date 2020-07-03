import math
import random

from simulation.utils.road.road import Road  # Definition of the road class
from simulation.utils.road.sections import (
    StraightRoad,
    LeftCircularArc,
    RightCircularArc,
)


def angle():
    return math.radians(random.random() * 180)


def radius():
    return random.random() * 3 + 1


road = Road()
road.append(StraightRoad(length=4))
road.append(LeftCircularArc(radius=radius(), angle=angle()))
road.append(StraightRoad(length=2))
road.append(RightCircularArc(radius=radius(), angle=angle()))
road.append(StraightRoad(length=2))
