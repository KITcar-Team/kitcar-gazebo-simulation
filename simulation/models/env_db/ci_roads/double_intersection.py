"""A road with two intersections."""

import math
import os

from simulation.utils.road.road import Road  # Definition of the road class
from simulation.utils.road.sections import Intersection, StraightRoad


def direction_str_to_int(env: str) -> int:
    """Convert a turn env variable to an Intersection attribute.

    This allows to specify "straight,left,right" as turn options through environment
    variables.
    """
    turn = os.environ.get(env, "straight")
    return getattr(Intersection, turn.upper())


def rule_str_to_int(env: str) -> int:
    """Convert a rule env variable to an Intersection attribute.

    This allows to specify "equal, yield, stop, priority_yield, priority_stop" as rule
    options through environment variables.
    """
    rule = os.environ.get(env, "equal")
    return getattr(Intersection, rule.upper())


# Read environment variables:
first_turn = direction_str_to_int("FIRST_TURN")
second_turn = direction_str_to_int("SECOND_TURN")
first_rule = rule_str_to_int("FIRST_RULE")
second_rule = rule_str_to_int("SECOND_RULE")
first_angle = math.radians(float(os.environ.get("FIRST_ANGLE", 90)))
second_angle = math.radians(float(os.environ.get("SECOND_ANGLE", 90)))

distance = float(os.environ.get("INTERSECTION_DISTANCE", 1))


"""Definition of the actual road.

Just a straight beginning and then two intersections with a short straight in between.
"""
road = Road()
road.append(StraightRoad(length=2))
road.append(Intersection(turn=first_turn, angle=first_angle, rule=first_rule))
road.append(StraightRoad(length=distance))
road.append(Intersection(turn=second_turn, angle=second_angle, rule=second_rule))
road.append(StraightRoad(length=2))
