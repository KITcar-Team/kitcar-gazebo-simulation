"""States used in the LaneStateMachine."""

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_evaluation.msg import State as StateMsg
from simulation.src.simulation_evaluation.src.state_machine.state_machines.state_machine import (
    StateMachine,
)

from .state import State


class FailureCollision(State):
    """This end state occurs when the car crashed into an obstacle.

    Once the state machine receives this state, the state can no longer change into a new one.

    Inheriting from State gives this class the ability to hand down description and value to State.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(description="Car crashed.", value=StateMsg.COLLISION)


class FailureBlockedArea(State):
    """This end state occurs when the car drives onto a blocked area."""

    def __init__(self):
        super().__init__(
            description="Car drove onto a blocked area.", value=StateMsg.BLOCKED_AREA
        )


class FailureOffRoad(State):
    """This end state occurs when the car drives of the road.

    Once the state machine receives this state, the state can no longer change into a new one.

    Inheriting from State gives this class the ability to hand down description and value to State.
    """

    def __init__(self):
        """Init end state.

        Initializing does not need any arguments however description and value have to initialized to super.
        """
        super().__init__(description="Car drove off the road.", value=StateMsg.OFF_ROAD)


class LaneState(State):
    """State that recognizes off road/collision failures."""

    def next(self, state_machine: StateMachine, input_msg: int):
        """Next state.

        Arguments:
            state_machine (StateMachine): On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            Class object of next failure state or defaults to self if no failure state was detected.
        """
        if input_msg == SpeakerMsg.OFF_ROAD:
            return state_machine.off_road
        elif input_msg == SpeakerMsg.COLLISION:
            return state_machine.collision
        elif input_msg == SpeakerMsg.BLOCKED_AREA:
            return state_machine.blocked_area

        return super().next(state_machine, input_msg)


class Right(LaneState):
    """The car drives in the right lane."""

    def __init__(self):
        """Set description and state value."""
        super().__init__(
            description="The car is in the right lane.", value=StateMsg.RIGHT_LANE,
        )

    def next(self, state_machine, input_msg: int) -> State:
        """Return new state."""
        if input_msg == SpeakerMsg.LEFT_LANE:
            return state_machine.left
        elif input_msg == SpeakerMsg.PARKING_LOT:
            return state_machine.parking_right

        return super().next(state_machine, input_msg)


class ParkingRight(LaneState):
    """The car parks on the right side."""

    def __init__(self):
        """Set description and state value."""
        super().__init__(
            description="The car parks on the right side.", value=StateMsg.PARKING_RIGHT,
        )

    def next(self, state_machine, input_msg: int):
        """Return new state."""
        if input_msg == SpeakerMsg.RIGHT_LANE:
            return state_machine.right

        return super().next(state_machine, input_msg)


class Left(LaneState):
    """The car drives in the left lane."""

    def __init__(self):
        """Set description and state value."""
        super().__init__(
            description="Car is in the left lane.", value=StateMsg.LEFT_LANE,
        )

    def next(self, state_machine, input_msg: int):
        """Return new state."""
        if input_msg == SpeakerMsg.RIGHT_LANE:
            return state_machine.right
        elif input_msg == SpeakerMsg.PARKING_LOT:
            return state_machine.parking_left

        return super().next(state_machine, input_msg)


class ParkingLeft(LaneState):
    """The car parks on the left side."""

    def __init__(self):
        """Set description and state value."""
        super().__init__(
            description="The car parks on the left side.", value=StateMsg.PARKING_LEFT,
        )

    def next(self, state_machine, input_msg: int):
        """Return new state."""
        if input_msg == SpeakerMsg.LEFT_LANE:
            return state_machine.left

        return super().next(state_machine, input_msg)
