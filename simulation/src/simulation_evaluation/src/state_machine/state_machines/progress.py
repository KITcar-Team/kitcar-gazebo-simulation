"""ProgressStateMachine keeps track if the car has started, is driving or has finished the
drive.

See :mod:`simulation.src.simulation_evaluation.src.state_machine.states.progress` for
implementation details of the states used in this StateMachine.
"""

from typing import Callable

from simulation.src.simulation_evaluation.src.state_machine.states.progress import (
    BeforeStart,
    Finished,
    Running,
)

from .state_machine import StateMachine


class ProgressStateMachine(StateMachine):
    """Keep track if the car has started, is driving or has finished the drive."""

    before_start: "State" = BeforeStart()  # noqa: F821
    """The car stands in front of the start line"""
    running: "State" = Running()  # noqa: F821
    """The car has started to drive"""
    finished: "State" = Finished()  # noqa: F821
    """The car finished the drive"""

    def __init__(self, callback: Callable[[], None]):
        """Initialize ProgressStateMachine.

        Arguments:
            callback: Function which gets executed when the state changes
        """
        super(ProgressStateMachine, self).__init__(
            state_machine=self.__class__,
            initial_state=ProgressStateMachine.before_start,
            callback=callback,
        )
