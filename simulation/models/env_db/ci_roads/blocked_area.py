from simulation.utils.road.road import Road
from simulation.utils.road.sections import BlockedArea, StraightRoad

road = Road()
road.append(StraightRoad(length=3))
road.append(BlockedArea())
road.append(StraightRoad(length=2))
