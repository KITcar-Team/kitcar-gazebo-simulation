"""The default road when launching this simulation."""

from simulation.utils.road.road import Road  # Definition of the road class
from simulation.utils.road.sections import StraightRoad, ZebraCrossing

road = Road()
road.append(StraightRoad(length=3))
road.append(ZebraCrossing(length=0.4))
road.append(StraightRoad(length=3))
road.append(ZebraCrossing(length=0.45))
road.append(StraightRoad(length=1))
