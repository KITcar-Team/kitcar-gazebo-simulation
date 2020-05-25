#!/usr/bin/env python3
import unittest

from simulation_evaluation.msg import State as StateMsg
from simulation_evaluation.msg import Referee as RefereeMsg

from simulation.src.simulation_evaluation.src.referee.referee import (
    Referee,
    StateMachineConnector,
)


class ModuleTest(unittest.TestCase):
    def setUp(self):
        """Create state machine connectors and referee."""
        self.prog_connector = StateMachineConnector(0, self.set_prog)
        self.overtaking_connector = StateMachineConnector(0, self.set_ot)
        self.parking_connector = StateMachineConnector(0, self.set_parking)
        self.priority_connector = StateMachineConnector(0, self.set_priority)

        self.referee = Referee(
            self.prog_connector,
            self.overtaking_connector,
            self.parking_connector,
            self.priority_connector,
        )

    def test_referee_prog(self):
        """Test if referee behaves as expected for changes in progress state."""
        self.prog_connector.state = StateMsg.PROGRESS_BEFORE_START
        self.assertEqual(self.referee.state, RefereeMsg.READY)
        self.referee.update(time=0, distance=0)
        self.assertEqual(self.referee.state, RefereeMsg.READY)

        self.prog_connector.state = StateMsg.PROGRESS_RUNNING
        self.referee.update(time=0, distance=0)
        self.assertEqual(self.referee.state, RefereeMsg.DRIVING)

        self.prog_connector.state = StateMsg.PROGRESS_FINISHED
        self.referee.update(time=0, distance=0)
        self.assertEqual(self.referee.state, RefereeMsg.COMPLETED)

    def test_referee_overtaking_parking(self):
        """Test if referee behaves as expected for changes in overtaking and parking states."""
        self.prog_connector.state = StateMsg.PROGRESS_RUNNING

        self.overtaking_connector.state = StateMsg.FORBIDDEN_LEFT
        self.referee.update(time=0, distance=0)
        # As only the overtaking state machine is in a forbidden state, it should be reset and the drive continued!
        self.assertEqual(self.ot_state, StateMsg.OVERTAKING_BEFORE_START)

        self.overtaking_connector.state = StateMsg.OVERTAKING_BEFORE_START
        self.parking_connector.state = StateMsg.FORBIDDEN_LEFT
        self.referee.update(time=0, distance=0)
        # As only the overtaking state machine is in a forbidden state, it should be reset and the drive continued!
        self.assertEqual(self.parking_state, StateMsg.PARKING_BEFORE_START)
        self.assertEqual(self.referee.state, RefereeMsg.DRIVING)

        self.overtaking_connector.state = StateMsg.COLLISION
        self.parking_connector.state = StateMsg.COLLISION
        self.priority_connector.state = StateMsg.COLLISION
        self.referee.update(time=0, distance=0)
        # As all state machines are in a forbidden state, the drive has failed
        self.assertEqual(self.referee.state, RefereeMsg.FAILED)

    def test_priority(self):
        self.prog_connector.state = StateMsg.PROGRESS_RUNNING

        self.priority_connector.state = StateMsg.PRIORITY_FORBIDDEN_IN_STOP_ZONE
        self.referee.update(time=0, distance=0)
        # As only the overtaking state machine is in a forbidden state, it should be reset and the drive continued!
        self.assertEqual(self.referee.state, RefereeMsg.FAILED)

    def set_parking(self, state):
        self.parking_state = state

    def set_ot(self, state):
        self.ot_state = state

    def set_prog(self, state):
        self.prog_state = state

    def set_priority(self, state):
        self.priority_state = state


if __name__ == "__main__":
    unittest.main()
