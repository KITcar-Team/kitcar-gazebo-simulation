"""A simple road for the onboarding task."""

from simulation.utils.road.road import Road  # Definition of the road class
from simulation.utils.road.sections import StraightRoad
from simulation.utils.road.sections import Intersection

import os
import math

road = Road()
road.append(StraightRoad(length=4))

# Read turn / angle / rule from environment variables
turn = int(os.environ.get("INTERSECTION_TURN", Intersection.STRAIGHT))
angle = math.radians(float(os.environ.get("INTERSECTION_ANGLE", 90)))
rule = int(os.environ.get("INTERSECTION_RULE", Intersection.EQUAL))

road.append(Intersection(turn=turn, angle=angle, rule=rule))
road.append(StraightRoad(length=4))
