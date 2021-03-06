"""States used in the PriorityStateMachine."""

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_evaluation.msg import State as StateMsg

from ..state_machines.state_machine import StateMachine
from .state import State


class Off(State):
    """This state is the default state.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car is not inside a stop zone.",
            value=StateMsg.PRIORITY_BEFORE_START,
        )

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine: On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here,
            check for failure state before returning this state.
        """
        if input_msg == SpeakerMsg.STOP_ZONE:
            return state_machine.in_stop_zone
        if input_msg == SpeakerMsg.HALT_ZONE:
            return state_machine.in_halt_zone

        return super().next(state_machine, input_msg)


class InHaltZone(State):
    """This state occurs when the car drives inside a halt zone.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car is inside a halt zone.", value=StateMsg.PRIORITY_IN_HALT_ZONE
        )

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine: On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here,
            check for failure state before returning this state.
        """
        if input_msg == SpeakerMsg.SPEED_HALTED:
            return state_machine.successfully_stopped

        if input_msg == SpeakerMsg.NO_STOP_ZONE:
            return state_machine.failure_in_stop_zone

        return super().next(state_machine, input_msg)


class InStopZone(State):
    """This state occurs when the car drives into a stop zone.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car is inside a stop zone.", value=StateMsg.PRIORITY_IN_STOP_ZONE
        )

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine: On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here,
            check for failure state before returning this state.
        """
        if input_msg == SpeakerMsg.SPEED_STOPPED:
            return state_machine.successfully_stopped
        if input_msg == SpeakerMsg.NO_STOP_ZONE:
            return state_machine.failure_in_stop_zone

        return super().next(state_machine, input_msg)


class SuccessfullyStopped(State):
    """This state occurs when the car stops in the stop zone.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car successfully stopped.", value=StateMsg.PRIORITY_SUCCESS_STOP
        )

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine: On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here,
            check for failure state before returning this state.
        """
        if input_msg == SpeakerMsg.NO_STOP_ZONE:
            return state_machine.off

        return super().next(state_machine, input_msg)


class FailureInStopZone(State):
    """This end state occurs when the car does not stop inside the stop zone.

    Once the state machine receives this state, the state can no longer change into a new
    one.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car drove out of stop zone but was not allowed to.",
            value=StateMsg.PRIORITY_FORBIDDEN_IN_STOP_ZONE,
        )
