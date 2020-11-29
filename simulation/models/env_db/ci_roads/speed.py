import os

from simulation.utils.road.road import Road
from simulation.utils.road.sections import StraightRoad
from simulation.utils.road.sections.speed_limit import SpeedLimit

speed_limit = int(os.environ.get("SPEED_LIMIT", 30))

limit = SpeedLimit(2, speed_limit)

road = Road()
road.append(StraightRoad(length=4, speed_limits=[limit]))
