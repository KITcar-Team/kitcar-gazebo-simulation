# -*- coding: utf-8 -*-
"""States used in the ActiveState class."""

from simulation_evaluation.msg import State as StateMsg

from .state import State

__copyright__ = "KITcar"


class FailureCollision(State):
    """This end state occurs when the car crashed into an obstacle.

    Once the state machine receives this state, the state can no longer change into a new one.

    Inheriting from State gives this class the ability to hand down description and value to State.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(description="Car crashed.", value=StateMsg.COLLISION)


class FailureOffRoad(State):
    """This end state occurs when the car drives of the road.

    Once the state machine receives this state, the state can no longer change into a new one.

    Inheriting from State gives this class the ability to hand down description and value to State.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(description="Car drove off the road.", value=StateMsg.OFF_ROAD)


class FailureLeft(State):
    """This end state occurs when the car drives onto the left lane.

    Once the state machine receives this state, the state can no longer change into a new one.

    Inheriting from State gives this class the ability to hand down description and value to State.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(
            description="Car drove onto the left lane.", value=StateMsg.FORBIDDEN_LEFT
        )
