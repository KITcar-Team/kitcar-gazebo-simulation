# -*- coding: utf-8 -*-
"""States used in the PriorityStateMachine."""

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_evaluation.msg import State as StateMsg
from simulation.src.simulation_evaluation.src.state_machine.state_machines.state_machine import (
    StateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.states.active import ActiveState
from simulation.src.simulation_evaluation.src.state_machine.states.state import State

__copyright__ = "KITcar"


class Off(ActiveState):
    """This state is the default state.

    Once the state machine receives this state, the next state will we chage accordingly to its next method.

    Inheriting from ActiveState gives this class the ability to hand down description and value to ActiveState. Same \
        goes for input_msg which gets parsed to the method next if no state change was detected.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(
            description="Car is not inside a stop zone.",
            value=StateMsg.PRIORITY_BEFORE_START,
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
        if input_msg == SpeakerMsg.STOP_ZONE:
            return state_machine.in_stop_zone
        if input_msg == SpeakerMsg.HALT_ZONE:
            return state_machine.in_halt_zone

        return super().next(state_machine, input_msg)


class InHaltZone(ActiveState):
    """This state occurs when the car drives inside a halt zone.

    Once the state machine receives this state, the next state will we chage accordingly to its next method.

    Inheriting from ActiveState gives this class the ability to hand down description and value to ActiveState. Same \
        goes for input_msg which gets parsed to the method next if no state change was detected.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(
            description="Car is inside a halt zone.", value=StateMsg.PRIORITY_IN_HALT_ZONE
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
        if input_msg == SpeakerMsg.SPEED_HALTED:
            return state_machine.off

        if input_msg == SpeakerMsg.NO_STOP_ZONE:
            return state_machine.failure_in_stop_zone

        return super().next(state_machine, input_msg)


class InStopZone(ActiveState):
    """This state occurs when the car drives into a stop zone.

    Once the state machine receives this state, the next state will we chage accordingly to its next method.

    Inheriting from ActiveState gives this class the ability to hand down description and value to ActiveState. Same \
        goes for input_msg which gets parsed to the method next if no state change was detected.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(
            description="Car is inside a stop zone.", value=StateMsg.PRIORITY_IN_STOP_ZONE
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
        if input_msg == SpeakerMsg.SPEED_STOPPED:
            return state_machine.successfully_stopped
        if input_msg == SpeakerMsg.NO_STOP_ZONE:
            return state_machine.failure_in_stop_zone

        return super().next(state_machine, input_msg)


class SuccessfullyStopped(ActiveState):
    """This state occurs when the car stops in the stop zone.

    Once the state machine receives this state, the next state will we chage accordingly to its next method.

    Inheriting from ActiveState gives this class the ability to hand down description and value to ActiveState. Same \
        goes for input_msg which gets parsed to the method next if no state change was detected.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(
            description="Car successfully stopped.", value=StateMsg.PRIORITY_SUCCESS_STOP
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
        if input_msg == SpeakerMsg.NO_STOP_ZONE:
            return state_machine.off

        return super().next(state_machine, input_msg)


class FailureInStopZone(State):
    """This end state occurs when the car does not stop inside the stop zone.

    Once the state machine receives this state, the state can no longer change into a new one.

    Inheriting from State gives this class the ability to hand down description and value to State.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(
            description="Car drove out of stop zone but was not allowed to.",
            value=StateMsg.PRIORITY_FORBIDDEN_IN_STOP_ZONE,
        )
