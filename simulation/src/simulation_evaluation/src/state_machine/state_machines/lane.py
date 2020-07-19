"""LaneStateMachine keeps track of where the car drives.

See :mod:`simulation.src.simulation_evaluation.src.state_machine.states.lane`
for implementation details of the states used in this StateMachine.
"""

from typing import Callable

from simulation.src.simulation_evaluation.src.state_machine.state_machines.state_machine import (
    StateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.states.lane import (
    Left,
    Right,
    ParkingLeft,
    ParkingRight,
    FailureCollision,
    FailureOffRoad,
)


class LaneStateMachine(StateMachine):
    """Keep track of which part of the road the car is on.

    .. autoattribute:: StateMachine.failure_collision
    .. autoattribute:: StateMachine.failure_off_road
    """

    collision = FailureCollision()
    """End state when driving into an obstacle"""
    off_road = FailureOffRoad()
    """End state when driving of the road"""
    right = Right()  # noqa: F821
    """The car is in the right lane."""
    left: "ActiveState" = Left()  # noqa: F821
    """The car is in the left lane."""
    parking_right: "ActiveState" = ParkingRight()  # noqa: F821
    """The car is parking on the right side."""
    parking_left: "ActiveState" = ParkingLeft()  # noqa: F821
    """The car is parking on the left side."""

    def __init__(self, callback: Callable[[], None]):
        """Initialize OvertakingStateMachine.

        Arguments:
            callback: Function which gets executed when the state changes
        """
        super(LaneStateMachine, self).__init__(
            state_machine=self.__class__,
            initial_state=LaneStateMachine.right,
            callback=callback,
        )
