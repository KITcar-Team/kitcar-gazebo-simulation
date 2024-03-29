import os

from simulation.utils.road.road import Road
from simulation.utils.road.sections import StraightRoad

speed_limit = int(os.environ.get("SPEED_LIMIT", 30))

road = Road()
s = StraightRoad(length=8)
s.add_speed_limit(4, speed_limit)
road.append(s)
