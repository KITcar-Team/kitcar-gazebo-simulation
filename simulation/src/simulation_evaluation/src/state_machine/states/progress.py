"""States used in the ProgressStateMachine."""

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_evaluation.msg import State as StateMsg

from ..state_machines.state_machine import StateMachine
from .state import State


class BeforeStart(State):
    """This state is the default state.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car has not started driving yet.",
            value=StateMsg.PROGRESS_BEFORE_START,
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
        if input_msg == SpeakerMsg.DRIVING_ZONE:
            return state_machine.running

        return super().next(state_machine, input_msg)


class Running(State):
    """This state occurs when the drive has started.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(description="Car is driving.", value=StateMsg.PROGRESS_RUNNING)

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine: On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here,
            check for failure state before returning this state.
        """
        if input_msg == SpeakerMsg.END_ZONE:
            return state_machine.finished

        return super().next(state_machine, input_msg)


class Finished(State):
    """This state occurs when the drive has finished.

    Once the state machine receives this state, the state can no longer change into a new
    one.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car has finished driving.", value=StateMsg.PROGRESS_FINISHED
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
        return super().next(state_machine, input_msg)
