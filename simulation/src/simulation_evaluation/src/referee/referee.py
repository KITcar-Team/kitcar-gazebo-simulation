"""Definition of the Referee class.

The referee is used to provide a score and other metrics to a simulated drive.
"""

from typing import Callable
from dataclasses import dataclass

from simulation_evaluation.msg import Referee as RefereeMsg
from simulation_evaluation.msg import State as StateMsg


@dataclass
class StateMachineConnector:
    """Helper class to store states and manipulate corresponding state machine.

    Args:
        state: State of the state machine.
        set_state: Call to change the state of the state_machine.
    """

    previous_state: int
    """State the state machine was in before the current state."""
    _state: int
    set_state: Callable[[int], None]
    """Change the state of the state_machine."""

    def __init__(self, state: int, set_state: Callable[[int], None]):
        self.previous_state = state
        self._state = state
        self.set_state = set_state

    @property
    def state(self):
        """State of the state machine."""
        return self._state

    @state.setter
    def state(self, val: int):
        self.previous_state = self._state
        self._state = val


@dataclass
class Observation:
    """Track what happens while driving and calculate a score."""

    start_time: float = 0
    """Time [s] when the car left the start zone."""
    current_time: float = 0
    """Current time [s]."""

    start_distance: float = 0
    """Distance [m] already driven in start zone."""
    current_distance: float = 0
    """Distance [m] already driven."""

    multiplicator: float = 1
    """Multiplicator used to calculate the score."""
    mistakes: float = 0
    """Mistakes [point] made by the car."""
    parking_successes: int = 0
    """Number of successful parking attempts."""

    @property
    def score(self) -> float:
        """Score calculated from other attributes."""
        return self.distance * self.multiplicator - self.mistakes

    @property
    def distance(self) -> float:
        """Difference between start and current distance."""
        return self.current_distance - self.start_distance

    @property
    def duration(self) -> float:
        """Difference between start and current time."""
        return self.current_time - self.start_time


class Referee:
    """Class to evaluate a drive by keeping track of the state_machines.

    Args:
        progress: Connector to progress state machine.
        overtaking: Connector to overtaking state machine.
        parking: Connector to parking state machine.
        priority: Connector to priority state machine.
        initial_observation: Referee's observation initialization value.
        reset_callback: Function to reset the referee.
    """

    def __init__(
        self,
        progress: StateMachineConnector,
        overtaking: StateMachineConnector,
        parking: StateMachineConnector,
        priority: StateMachineConnector,
        initial_observation: Observation = None,
    ):
        self.progress = progress
        self.overtaking = overtaking
        self.parking = parking
        self.priority = priority

        if initial_observation is None:
            initial_observation = Observation()
        self.observation = initial_observation

        self.state = RefereeMsg.READY

    def update(self, time: float, distance: float):
        """Update the referee's observation.

        Args:
            time: Current time in seconds.
            distance: Distance driven in meters.
        """
        if (
            self.progress.state == StateMsg.PROGRESS_RUNNING
            and self.state == RefereeMsg.READY
        ):
            # This means that the car has not started driving before!
            self.observation.start_time = time
            self.observation.start_distance = distance
            self.state = RefereeMsg.DRIVING
        elif self.progress.state == StateMsg.PROGRESS_FINISHED:
            self.state = RefereeMsg.COMPLETED

        # Remainder of function only needs to be called if the test is still running.
        if not self.state == RefereeMsg.DRIVING:
            return

        # Get successful parking
        if (
            self.parking.state == StateMsg.PARKING_SUCCESS
            and self.parking.previous_state != StateMsg.PARKING_SUCCESS
        ):
            self.parking.state = StateMsg.PARKING_SUCCESS
            self.observation.multiplicator += 1.0
            self.observation.parking_successes += 1
            print("PARKING COMPLETED SUCCESSFULLY (RECEIVED IN REFEREE)")

        self.observation.current_distance = distance  # Get current distance
        self.observation.current_time = time  # Get time in seconds

        if (
            self.overtaking.state < 0 and self.parking.state < 0 and self.priority.state < 0
        ) or self.priority.state == StateMsg.PRIORITY_FORBIDDEN_IN_STOP_ZONE:
            self.state = RefereeMsg.FAILED
            # return f"Referee going into failing state with overtaking state \
            #        {self.overtaking.state} and parking state {self.parking.state}"
            return
            # TODO Reset car to beginning of next section!

        if self.overtaking.state < 0:
            self.overtaking.set_state(StateMsg.OVERTAKING_BEFORE_START)
        if self.parking.state < 0:
            self.parking.set_state(StateMsg.PARKING_BEFORE_START)
        if self.priority.state < 0:
            self.priority.set_state(StateMsg.PRIORITY_BEFORE_START)

    def reset(self, initial_observation: Observation = None):
        """Reset referee observations, state and state machines."""

        if initial_observation is None:
            initial_observation = Observation()
        self.observation = initial_observation

        def reset_connector(con, state):
            con.set_state(state)
            con.state = state

        reset_connector(self.overtaking, StateMsg.OVERTAKING_BEFORE_START)
        reset_connector(self.parking, StateMsg.PARKING_BEFORE_START)
        reset_connector(self.priority, StateMsg.PRIORITY_BEFORE_START)
        reset_connector(self.progress, StateMsg.PROGRESS_BEFORE_START)

        self.state = RefereeMsg.READY
