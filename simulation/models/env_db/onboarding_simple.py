"""A simple road for the onboarding task."""

from simulation.utils.road.road import Road  # Definition of the road class
from simulation.utils.road.sections import StraightRoad
from simulation.utils.road.sections import Intersection


road = Road()
road.append(StraightRoad())
road.append(StraightRoad(length=2))
road.append(Intersection())
road.append(StraightRoad())
