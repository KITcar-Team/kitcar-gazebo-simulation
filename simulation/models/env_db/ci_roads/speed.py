import os

from simulation.utils.road.road import Road
from simulation.utils.road.sections import StraightRoad

speed_limit = int(os.environ.get("SPEED_LIMIT", 30))

road = Road()
s = road.append(StraightRoad(length=4))
s.add_speed_limit(2, speed_limit)
