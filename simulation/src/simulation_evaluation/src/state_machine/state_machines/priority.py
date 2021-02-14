"""PriorityStateMachine keeps track of stoping or halting in front of stop or halt lines.

See :mod:`simulation.src.simulation_evaluation.src.state_machine.states.priority` for
implementation details of the states used in this StateMachine.
"""

from typing import Callable

from simulation.src.simulation_evaluation.src.state_machine.states.priority import (
    FailureInStopZone,
    InHaltZone,
    InStopZone,
    Off,
    SuccessfullyStopped,
)

from .state_machine import StateMachine

__copyright__ = "KITcar"


class PriorityStateMachine(StateMachine):
    """Keep track of stoping and halting in front of stop or halt lines."""

    off: "State" = Off()  # noqa: F821
    """Default state"""
    in_stop_zone: "State" = InStopZone()  # noqa: F821
    """The car is inside a stop zone"""
    in_halt_zone: "State" = InHaltZone()  # noqa: F821
    """The car is inside a halt zone"""
    successfully_stopped: "State" = SuccessfullyStopped()  # noqa: F821
    """The car successfully stopes in the stop zone"""
    failure_in_stop_zone: "State" = FailureInStopZone()  # noqa: F821
    """End state when the car does not stop inside the stop zone"""

    def __init__(self, callback: Callable[[], None]):
        """Initialize PriorityStateMachine.

        Arguments:
            callback: Function which gets executed when the state changes
        """
        super().__init__(
            state_machine=self.__class__,
            initial_state=PriorityStateMachine.off,
            callback=callback,
        )
