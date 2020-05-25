# -*- coding: utf-8 -*-
"""ParkingStateMachine keeps track of parking.

See :mod:`simulation.src.simulation_evaluation.src.state_machine.states.parking` for implementation details of the \
    states used in this StateMachine.
"""

from typing import Callable

from simulation.src.simulation_evaluation.src.state_machine.state_machines.state_machine import (
    StateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.states.parking import (
    FailureInLeftLane,
    FailureInRightLane,
    InParkingZone,
    Off,
    Parking,
    ParkingAttempt,
    ParkingOut,
    SuccessfullyParked,
)

__copyright__ = "KITcar"


class ParkingStateMachine(StateMachine):
    """Keep track of parking.

    Inheriting from StateMachine makes it possible to handle all states from outside.

    .. autoattribute:: StateMachine.failure_collision
    .. autoattribute:: StateMachine.failure_off_road
    .. autoattribute:: StateMachine.failure_left
    """

    off: "ActiveState" = Off()  # noqa: F821
    """Default state"""
    in_parking_zone: "ActiveState" = InParkingZone()  # noqa: F821
    """The car is inside a parking zone"""
    parking_attempt: "ActiveState" = ParkingAttempt()  # noqa: F821
    """The car starts an attempt to park in"""
    parking: "ActiveState" = Parking()  # noqa: F821
    """The car drives into a parking space"""
    successfully_parked: "ActiveState" = SuccessfullyParked()  # noqa: F821
    """The car successfully parkes inside a parking space"""
    parking_out: "ActiveState" = ParkingOut()  # noqa: F821
    """The car drives out of the parkin space"""
    failure_in_right: "State" = FailureInRightLane()  # noqa: F821
    """End state when the car drives in the right lane when it's not allowed to"""
    failure_in_left: "State" = FailureInLeftLane()  # noqa: F821
    """End state when the car drives in the left lane when it's not allowed to"""

    def __init__(self, callback: Callable[[], None]):
        """Initialize ParkingStateMachine.

        Arguments:
            callback: Function which gets executed when the state changes
        """
        super(ParkingStateMachine, self).__init__(
            state_machine=self.__class__,
            initial_state=ParkingStateMachine.off,
            callback=callback,
        )
