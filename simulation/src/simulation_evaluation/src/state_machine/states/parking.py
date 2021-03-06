"""States used in the ParkingStateMachine."""

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_evaluation.msg import State as StateMsg

from ..state_machines.state_machine import StateMachine
from .state import State


class ParkingState(State):
    def next(self, state_machine, input_msg: int):
        """Return updated state."""
        if input_msg == SpeakerMsg.NO_PARKING_ZONE:
            return state_machine.off

        return super().next(state_machine, input_msg)


class Off(ParkingState):
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
            description="Car is not inside a parking zone.",
            value=StateMsg.PARKING_BEFORE_START,
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
        if input_msg == SpeakerMsg.PARKING_ZONE:
            return state_machine.in_parking_zone

        return super().next(state_machine, input_msg)


class InParkingZone(ParkingState):
    """This state occurs when the car drives into the parking zone.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car is inside a parking zone.", value=StateMsg.PARKING_IN_ZONE
        )

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine (StateMachine): On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here,
            check for failure state before returning this state.
        """
        if input_msg == SpeakerMsg.LEFT_LANE:
            return state_machine.parking_attempt
        if input_msg == SpeakerMsg.PARKING_SPOT:
            return state_machine.parking
        if input_msg == SpeakerMsg.NO_PARKING_ZONE:
            return state_machine.off

        return super().next(state_machine, input_msg)


class ParkingAttempt(ParkingState):
    """This state occurs when the car starts an attempt to park in.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car is driving into one parking spot or at least is trying to.",
            value=StateMsg.PARKING_ATTEMPT,
        )

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine (StateMachine): On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here,
            check for failure state before returning this state.
        """
        if input_msg == SpeakerMsg.LEFT_LANE:
            return state_machine.parking_attempt
        if input_msg == SpeakerMsg.PARKING_SPOT:
            return state_machine.parking
        if input_msg == SpeakerMsg.RIGHT_LANE:
            return state_machine.failure_in_right

        return super().next(state_machine, input_msg)


class Parking(ParkingState):
    """This state occurs when the car drives a parking space.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car is waiting until it has successfully parked.",
            value=StateMsg.PARKING_IN_SPOT,
        )

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine (StateMachine): On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here,
            check for failure state before
            returning this state.
        """
        if input_msg == SpeakerMsg.SPEED_0:
            return state_machine.successfully_parked
        if input_msg == SpeakerMsg.RIGHT_LANE:
            return state_machine.failure_in_right
        if input_msg == SpeakerMsg.LEFT_LANE:
            return state_machine.failure_in_left

        return super().next(state_machine, input_msg)


class SuccessfullyParked(ParkingState):
    """This state occurs when the car successfully parks inside a parking space.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car has successfully parked.", value=StateMsg.PARKING_SUCCESS
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
        if input_msg == SpeakerMsg.LEFT_LANE:
            return state_machine.parking_out
        if input_msg == SpeakerMsg.RIGHT_LANE:
            return state_machine.in_parking_zone

        return super().next(state_machine, input_msg)


class ParkingOut(ParkingState):
    """This state occurs when the car drives out of the parking space.

    Once the state machine receives this state, the next state will we chage accordingly to
    its next method.
    """

    def __init__(self):
        """Init state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car is driving out of the parking spot.",
            value=StateMsg.PARKING_OUT,
        )

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine (StateMachine): On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next state. If no state change was detected here,
            check for failure state before returning this state.
        """
        if input_msg == SpeakerMsg.LEFT_LANE:
            return state_machine.parking_out
        if input_msg == SpeakerMsg.RIGHT_LANE:
            return state_machine.in_parking_zone

        return super().next(state_machine, input_msg)


class FailureInRightLane(State):
    """This end state occurs when the car drives onto the right lane.

    Once the state machine receives this state, the state can no longer change into a new
    one.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car failed to park.", value=StateMsg.PARKING_FORBIDDEN_RIGHT
        )


class FailureInLeftLane(State):
    """This end state occurs when the car drives onto the left lane.

    Once the state machine receives this state, the state can no longer change into a new
    one.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to
        initialized to super.
        """
        super().__init__(
            description="Car did not halt in parking spot.",
            value=StateMsg.PARKING_FORBIDDEN_LEFT,
        )
