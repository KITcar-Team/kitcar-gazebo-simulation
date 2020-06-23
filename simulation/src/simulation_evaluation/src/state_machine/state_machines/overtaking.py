# -*- coding: utf-8 -*-
"""OvertakingStateMachine keeps track of overtaking obstacles.

See :mod:`simulation.src.simulation_evaluation.src.state_machine.states.overtaking` for implementation details of the \
    states used in this StateMachine.
"""

from typing import Callable

from simulation.src.simulation_evaluation.src.state_machine.state_machines.state_machine import (
    StateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.states.overtaking import (
    Left,
    Off,
    Right,
)

__copyright__ = "KITcar"


class OvertakingStateMachine(StateMachine):
    """Keep track of overtaking obstacles.

    Inheriting from StateMachine makes it possible to handle all states from outside.

    .. autoattribute:: StateMachine.failure_collision
    .. autoattribute:: StateMachine.failure_off_road
    .. autoattribute:: StateMachine.failure_left
    """

    off: "ActiveState" = Off()  # noqa: F821
    """Default state"""
    right: "ActiveState" = Right()  # noqa: F821
    """The car is inside the the overtaking zone and on the right line"""
    left: "ActiveState" = Left()  # noqa: F821
    """The car is inside the the overtaking zone and on the left line"""

    def __init__(self, callback: Callable[[], None]):
        """Initialize OvertakingStateMachine.

        Arguments:
            callback: Function which gets executed when the state changes
        """
        super(OvertakingStateMachine, self).__init__(
            state_machine=self.__class__,
            initial_state=OvertakingStateMachine.off,
            callback=callback,
        )
