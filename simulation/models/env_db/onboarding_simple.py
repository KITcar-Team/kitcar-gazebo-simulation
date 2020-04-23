"""A simple road for the onboarding task."""

from road.road import Road  # Definition of the road class
from road.sections import StraightRoad
from road.sections import Intersection


road = Road()
road.append(StraightRoad())
road.append(StraightRoad(length=2))
road.append(Intersection())
road.append(StraightRoad())
