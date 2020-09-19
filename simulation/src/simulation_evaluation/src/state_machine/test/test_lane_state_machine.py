"""ParkingStateMachine ModuleTest."""

import unittest

from simulation_evaluation.msg import Speaker as SpeakerMsg

from simulation.src.simulation_evaluation.src.state_machine.state_machines.lane import (
    LaneStateMachine,
)

from .test import Test


class ModuleTest(Test):
    """Test if the ParkingStateMachine behaves as expected."""

    def test_left_lane(self):
        """Test if state machine works as expected drives between the left and right
        lane."""
        inputs = [
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.LEFT_LANE,
        ]
        states = [
            LaneStateMachine.right,
            LaneStateMachine.left,
            LaneStateMachine.right,
            LaneStateMachine.left,
        ]

        self.state_machine_assert_on_input(
            LaneStateMachine(self.callback), inputs, states, 3
        )

    def test_parking_right(self):
        """Test if state machine works as expected if car parks right."""
        inputs = [
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.PARKING_LOT,
            SpeakerMsg.RIGHT_LANE,
        ]
        states = [
            LaneStateMachine.right,
            LaneStateMachine.parking_right,
            LaneStateMachine.right,
        ]

        self.state_machine_assert_on_input(
            LaneStateMachine(self.callback), inputs, states, 2
        )

    def test_parking_left(self):
        """Test if state machine works as expected if car parks left."""
        inputs = [
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.PARKING_LOT,
            SpeakerMsg.LEFT_LANE,
        ]
        states = [
            LaneStateMachine.right,
            LaneStateMachine.left,
            LaneStateMachine.parking_left,
            LaneStateMachine.left,
        ]

        self.state_machine_assert_on_input(
            LaneStateMachine(self.callback), inputs, states, 3
        )

    def test_collision(self):
        """Test if state machine works as expected if car collides."""
        inputs = [
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.COLLISION,
        ]
        states = [
            LaneStateMachine.right,
            LaneStateMachine.collision,
        ]

        self.state_machine_assert_on_input(
            LaneStateMachine(self.callback), inputs, states, 1
        )

    def test_off_road(self):
        """Test if state machine works as expected if car drives offroad while trying to
        park in."""
        inputs = [
            SpeakerMsg.SPEED_UNLIMITED_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.OFF_ROAD,
        ]
        states = [
            LaneStateMachine.right,
            LaneStateMachine.left,
            LaneStateMachine.off_road,
        ]

        self.state_machine_assert_on_input(
            LaneStateMachine(self.callback), inputs, states, 2
        )


if __name__ == "__main__":
    unittest.main()
