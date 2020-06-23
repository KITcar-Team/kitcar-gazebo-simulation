# -*- coding: utf-8 -*-
"""ActiveState checks for failure states."""

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation.src.simulation_evaluation.src.state_machine.state_machines.state_machine import (
    StateMachine,
)

from .state import State

__copyright__ = "KITcar"


class ActiveState(State):
    """ActiveState is an addition to State.

    When the metod next is called, it checks for failure states. Each failure state is an instance of State and \
        therefore is an end state.

    Inheriting from State gives this class the ability to hand down description and value to State. Same goes for \
        input_msg which gets parsed to the method next if no failure state was detected.

    Initializing does require the following arguments:

    Arguments:
        description (str): Human readable description of state
        value (int): Value of State Message (Defined in msg/State.msg)
    """

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next failure state.

        Arguments:
            state_machine (StateMachine): On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next failure state or defaults to self if no failure state was detected.
        """
        if input_msg == SpeakerMsg.OFF_ROAD:
            return state_machine.failure_off_road
        elif input_msg == SpeakerMsg.LEFT_LANE:
            return state_machine.failure_left
        elif input_msg == SpeakerMsg.COLLISION:
            return state_machine.failure_collision

        return super().next(state_machine, input_msg)
