"""ParkingStateMachine ModuleTest."""

import unittest

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation.src.simulation_evaluation.src.state_machine.state_machines.parking import (
    ParkingStateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.test.test import Test


__copyright__ = "KITcar"


class ModuleTest(Test):
    """Test if the ParkingStateMachine behaves as expected."""

    def test_correct_behavior(self):
        """Test if correct behavior works as expected."""
        inputs = [
            SpeakerMsg.SPEED_UNLIMITED_ZONE,
            SpeakerMsg.PARKING_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.PARKING_SPOT,
            SpeakerMsg.SPEED_HALTED,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.NO_PARKING_ZONE,
        ]
        states = [
            ParkingStateMachine.off,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.parking_attempt,
            ParkingStateMachine.parking,
            ParkingStateMachine.successfully_parked,
            ParkingStateMachine.parking_out,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.off,
        ]

        self.state_machine_assert_on_input(
            ParkingStateMachine(self.callback), inputs, states, 7
        )

    def test_input_mutliple_times(self):
        """Test if inputting speaker msgs multiple times changes anything (which it shouldn't)."""
        inputs = [
            SpeakerMsg.SPEED_UNLIMITED_ZONE,
            SpeakerMsg.SPEED_UNLIMITED_ZONE,
            SpeakerMsg.PARKING_ZONE,
            SpeakerMsg.PARKING_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.PARKING_SPOT,
            SpeakerMsg.PARKING_SPOT,
            SpeakerMsg.SPEED_HALTED,
            SpeakerMsg.SPEED_HALTED,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.NO_PARKING_ZONE,
            SpeakerMsg.NO_PARKING_ZONE,
        ]
        states = [
            ParkingStateMachine.off,
            ParkingStateMachine.off,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.parking_attempt,
            ParkingStateMachine.parking_attempt,
            ParkingStateMachine.parking,
            ParkingStateMachine.parking,
            ParkingStateMachine.successfully_parked,
            ParkingStateMachine.successfully_parked,
            ParkingStateMachine.parking_out,
            ParkingStateMachine.parking_out,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.off,
            ParkingStateMachine.off,
        ]

        self.state_machine_assert_on_input(
            ParkingStateMachine(self.callback), inputs, states, 7
        )

    def test_unrelated_msgs(self):
        """Test if inputting unrelated speaker msgs changes anything (which it shouldn't)."""
        inputs = [
            SpeakerMsg.SPEED_UNLIMITED_ZONE,
            SpeakerMsg.SPEED_51_60,
            SpeakerMsg.PARKING_ZONE,
            SpeakerMsg.STOP_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.OVERTAKING_ZONE,
            SpeakerMsg.PARKING_SPOT,
            SpeakerMsg.SPEED_10_ZONE,
            SpeakerMsg.SPEED_HALTED,
            SpeakerMsg.DRIVING_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.NO_OVERTAKING_ZONE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.SPEED_90_ZONE,
            SpeakerMsg.NO_PARKING_ZONE,
            SpeakerMsg.NO_OVERTAKING_ZONE,
        ]
        states = [
            ParkingStateMachine.off,
            ParkingStateMachine.off,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.parking_attempt,
            ParkingStateMachine.parking_attempt,
            ParkingStateMachine.parking,
            ParkingStateMachine.parking,
            ParkingStateMachine.successfully_parked,
            ParkingStateMachine.successfully_parked,
            ParkingStateMachine.parking_out,
            ParkingStateMachine.parking_out,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.off,
            ParkingStateMachine.off,
        ]

        self.state_machine_assert_on_input(
            ParkingStateMachine(self.callback), inputs, states, 7
        )

    def test_right_lane(self):
        """Test if state machine works as expected if car drives into right lane after trying to park in."""
        inputs = [
            SpeakerMsg.SPEED_UNLIMITED_ZONE,
            SpeakerMsg.PARKING_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.RIGHT_LANE,
        ]
        states = [
            ParkingStateMachine.off,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.parking_attempt,
            ParkingStateMachine.failure_in_right,
        ]

        self.state_machine_assert_on_input(
            ParkingStateMachine(self.callback), inputs, states, 3
        )

    def test_off_road(self):
        """Test if state machine works as expected if car drives offroad while trying to park in."""
        inputs = [
            SpeakerMsg.SPEED_UNLIMITED_ZONE,
            SpeakerMsg.PARKING_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.OFF_ROAD,
        ]
        states = [
            ParkingStateMachine.off,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.parking_attempt,
            ParkingStateMachine.failure_off_road,
        ]

        self.state_machine_assert_on_input(
            ParkingStateMachine(self.callback), inputs, states, 3
        )

    def test_left_lane(self):
        """Test if state machine works as expected if car drives left after parking in."""
        inputs = [
            SpeakerMsg.SPEED_UNLIMITED_ZONE,
            SpeakerMsg.PARKING_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.PARKING_SPOT,
            SpeakerMsg.SPEED_HALTED,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.PARKING_SPOT,
            SpeakerMsg.SPEED_HALTED,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.NO_PARKING_ZONE,
        ]
        states = [
            ParkingStateMachine.off,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.parking_attempt,
            ParkingStateMachine.parking,
            ParkingStateMachine.successfully_parked,
            ParkingStateMachine.parking_out,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.parking_attempt,
            ParkingStateMachine.parking,
            ParkingStateMachine.successfully_parked,
            ParkingStateMachine.parking_out,
            ParkingStateMachine.in_parking_zone,
            ParkingStateMachine.off,
        ]

        self.state_machine_assert_on_input(
            ParkingStateMachine(self.callback), inputs, states, 12
        )


if __name__ == "__main__":
    unittest.main()
