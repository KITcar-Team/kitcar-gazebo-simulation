"""States used in the OvertakingStateMachine."""

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_evaluation.msg import State as StateMsg

from ..state_machines.state_machine import StateMachine
from .state import State


class OvertakingState(State):
    def next(self, state_machine, input_msg: int):
        """Return updated state."""
        if input_msg == SpeakerMsg.NO_OVERTAKING_ZONE:
            return state_machine.off

        return super().next(state_machine, input_msg)


class Off(OvertakingState):
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
            description="Car is not inside an overtaking zone.",
            value=StateMsg.OVERTAKING_BEFORE_START,
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
        if input_msg == SpeakerMsg.OVERTAKING_ZONE:
            return state_machine.right

        return super().next(state_machine, input_msg)


class Right(OvertakingState):
    """This state occurs when the car drives into the overtaking zone and is on the right
    line.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car is inside an overtaking zone, on the right line.",
            value=StateMsg.OVERTAKING_RIGHT,
        )

    def next(self, state_machine, input_msg: int):
        """Next state.

        Arguments:
            state_machine: On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here,
            check for failure state before returning this state.
        """
        if input_msg == SpeakerMsg.LEFT_LANE:
            return state_machine.left

        return super().next(state_machine, input_msg)


class Left(OvertakingState):
    """This state occurs when the car is in the overtaking zone and in the left line.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car is inside an overtaking zone, on the left line.",
            value=StateMsg.OVERTAKING_LEFT,
        )

    def next(self, state_machine, input_msg: int):
        """Next state.

        Arguments:
            state_machine (StateMachine): On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here,
            check for failure state before returning this state.
        """
        if input_msg == SpeakerMsg.RIGHT_LANE:
            return state_machine.right

        return super().next(state_machine, input_msg)
