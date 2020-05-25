# -*- coding: utf-8 -*-
"""States used in the ProgressStateMachine."""

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_evaluation.msg import State as StateMsg
from simulation.src.simulation_evaluation.src.state_machine.state_machines.state_machine import (
    StateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.states.state import State

__copyright__ = "KITcar"


class BeforeStart(State):
    """This state is the default state.

    Once the state machine receives this state, the next state will we chage accordingly to its next method.

    Inheriting from ActiveState gives this class the ability to hand down description and value to ActiveState. Same \
        goes for input_msg which gets parsed to the method next if no state change was detected.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(
            description="Car has not started driving yet.",
            value=StateMsg.PROGRESS_BEFORE_START,
        )

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine (StateMachine): On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here, check for failure state before
            returning this state.
        """
        # Throw an error when car goes from start to end directly.
        assert input_msg != SpeakerMsg.END_ZONE

        if input_msg == SpeakerMsg.DRIVING_ZONE:
            return state_machine.running

        return super().next(state_machine, input_msg)


class Running(State):
    """This state occurs when the drive has started.

    Once the state machine receives this state, the next state will we chage accordingly to its next method.

    Inheriting from ActiveState gives this class the ability to hand down description and value to ActiveState. Same \
        goes for input_msg which gets parsed to the method next if no state change was detected.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(description="Car is driving.", value=StateMsg.PROGRESS_RUNNING)

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine (StateMachine): On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here, check for failure state before
            returning this state.
        """
        # Throw an error when car goes into start zone again. (Will throw on round trips on purpose)
        assert input_msg != SpeakerMsg.START_ZONE

        if input_msg == SpeakerMsg.END_ZONE:
            return state_machine.finished

        return super().next(state_machine, input_msg)


class Finished(State):
    """This state occurs when the drive has finished.

    Once the state machine receives this state, the state can no longer change into a new one.

    Inheriting from ActiveState gives this class the ability to hand down description and value to ActiveState. Same \
        goes for input_msg which gets parsed to the method next if no state change was detected.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(
            description="Car has finished driving.", value=StateMsg.PROGRESS_FINISHED
        )

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine (StateMachine): On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here, check for failure state before
            returning this state.
        """
        return super().next(state_machine, input_msg)
