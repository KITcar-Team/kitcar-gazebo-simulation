import math
import os
import random

from simulation.utils.road.road import Road
from simulation.utils.road.sections import LeftCircularArc, RightCircularArc, StraightRoad

speed_limit_1 = int(os.environ.get("SPEED_LIMIT_1", 20))
speed_limit_2 = int(os.environ.get("SPEED_LIMIT_2", 40))

angle = math.radians(random.random() * 180 + 36)
radius = random.random() * 3 + 1

road = Road()
road.append(StraightRoad(length=4))
left_arc = LeftCircularArc(
    radius=radius,
    angle=angle,
)

road.append(left_arc)
left_arc.add_speed_limit(arc_length=radius * angle / 2, speed=speed_limit_1)
right_arc = RightCircularArc(
    radius=radius,
    angle=angle,
)
road.append(right_arc)
right_arc.add_speed_limit(arc_length=radius * angle / 2, speed=speed_limit_2)
road.append(StraightRoad(length=4))
