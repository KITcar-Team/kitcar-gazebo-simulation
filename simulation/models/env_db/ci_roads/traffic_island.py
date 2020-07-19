from simulation.utils.road.road import Road
from simulation.utils.road.sections import StraightRoad, TrafficIsland

road = Road()
road.append(StraightRoad(length=3))
road.append(TrafficIsland(zebra_marking_type=TrafficIsland.LINES))
road.append(StraightRoad(length=2))
