import math
import os

from simulation.utils.road.road import Road
from simulation.utils.road.sections import Intersection, StraightRoad

speed_limit = int(os.environ.get("SPEED_LIMIT", 30))

road = Road()
road.append(StraightRoad(length=4))
road.append(Intersection(turn=Intersection.RIGHT, angle=math.radians(90)))
s = road.append(StraightRoad(length=4))
s.add_speed_limit(0, speed=speed_limit)
