"""A road with two intersections."""

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


# Read environment variables:
first_turn = direction_str_to_int("FIRST_TURN")
second_turn = direction_str_to_int("SECOND_TURN")

"""Definition of the actual road.

Just a straight beginning and then two intersections.
"""
road = Road()
road.append(StraightRoad(length=2))
road.append(Intersection(turn=first_turn, rule=Intersection.STOP))
road.append(StraightRoad(length=1))
road.append(Intersection(turn=second_turn, rule=Intersection.YIELD))
road.append(StraightRoad(length=2))
