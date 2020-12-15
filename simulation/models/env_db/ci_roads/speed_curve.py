import math
import os
import random

from simulation.utils.road.road import Road
from simulation.utils.road.sections import LeftCircularArc, RightCircularArc, StraightRoad
from simulation.utils.road.sections.speed_limit import SpeedLimit

speed_limit_1 = int(os.environ.get("SPEED_LIMIT_1", 20))
speed_limit_2 = int(os.environ.get("SPEED_LIMIT_2", 40))

angle = math.radians(random.random() * 180 + 36)
radius = random.random() * 3 + 1

road = Road()
road.append(StraightRoad(length=4))
road.append(
    LeftCircularArc(
        radius=radius,
        angle=angle,
        speed_limits=[SpeedLimit(radius * angle / 2, speed_limit_1)],
    )
)
road.append(
    RightCircularArc(
        radius=radius,
        angle=angle,
        speed_limits=[SpeedLimit(radius * angle / 2, speed_limit_2)],
    )
)
road.append(StraightRoad(length=4))
