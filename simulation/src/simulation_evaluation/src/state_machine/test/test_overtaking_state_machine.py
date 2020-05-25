"""OvertakingStateMachine ModuleTest."""

import unittest

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation.src.simulation_evaluation.src.state_machine.state_machines.overtaking import (
    OvertakingStateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.test.test import Test


__copyright__ = "KITcar"


class ModuleTest(Test):
    """Test if the OvertakingStateMachine behaves as expected."""

    def test_correct_behavior(self):
        """Test if correct behavior works as expected."""
        inputs = [
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.OVERTAKING_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.NO_OVERTAKING_ZONE,
        ]
        states = [
            OvertakingStateMachine.off,
            OvertakingStateMachine.right,
            OvertakingStateMachine.left,
            OvertakingStateMachine.right,
            OvertakingStateMachine.off,
        ]

        self.state_machine_assert_on_input(
            OvertakingStateMachine(self.callback), inputs, states, 4
        )

    def test_input_mutliple_times(self):
        """Test if inputting speaker msgs multiple times changes anything (which it shouldn't)."""
        inputs = [
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.OVERTAKING_ZONE,
            SpeakerMsg.OVERTAKING_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.NO_OVERTAKING_ZONE,
            SpeakerMsg.NO_OVERTAKING_ZONE,
        ]
        states = [
            OvertakingStateMachine.off,
            OvertakingStateMachine.off,
            OvertakingStateMachine.right,
            OvertakingStateMachine.right,
            OvertakingStateMachine.left,
            OvertakingStateMachine.left,
            OvertakingStateMachine.right,
            OvertakingStateMachine.right,
            OvertakingStateMachine.off,
            OvertakingStateMachine.off,
        ]

        self.state_machine_assert_on_input(
            OvertakingStateMachine(self.callback), inputs, states, 4
        )

    def test_unrelated_msgs(self):
        """Test if inputting unrelated speaker msgs changes anything (which it shouldn't)."""
        inputs = [
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.SPEED_0,
            SpeakerMsg.OVERTAKING_ZONE,
            SpeakerMsg.DRIVING_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.NO_STOP_ZONE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.PARKING_ZONE,
            SpeakerMsg.HALT_ZONE,
            SpeakerMsg.NO_OVERTAKING_ZONE,
        ]
        states = [
            OvertakingStateMachine.off,
            OvertakingStateMachine.off,
            OvertakingStateMachine.right,
            OvertakingStateMachine.right,
            OvertakingStateMachine.left,
            OvertakingStateMachine.left,
            OvertakingStateMachine.right,
            OvertakingStateMachine.right,
            OvertakingStateMachine.right,
            OvertakingStateMachine.off,
        ]

        self.state_machine_assert_on_input(
            OvertakingStateMachine(self.callback), inputs, states, 4
        )

    def test_left_side(self):
        """Test if state machine reacts as expected if car is still on left side after overtaking zone ends."""
        inputs = [
            SpeakerMsg.OVERTAKING_ZONE,
            SpeakerMsg.LEFT_LANE,
            SpeakerMsg.NO_OVERTAKING_ZONE,
        ]
        states = [
            OvertakingStateMachine.right,
            OvertakingStateMachine.left,
            OvertakingStateMachine.failure_left,
        ]

        self.state_machine_assert_on_input(
            OvertakingStateMachine(self.callback), inputs, states, 3
        )

    def test_off_road(self):
        """Test if state machine reacts as expected if car is driving off road."""
        inputs = [SpeakerMsg.OFF_ROAD]
        states = [OvertakingStateMachine.failure_off_road]

        self.state_machine_assert_on_input(
            OvertakingStateMachine(self.callback), inputs, states, 1
        )

        inputs = [SpeakerMsg.OVERTAKING_ZONE, SpeakerMsg.OFF_ROAD]
        states = [OvertakingStateMachine.right, OvertakingStateMachine.failure_off_road]

        self.state_machine_assert_on_input(
            OvertakingStateMachine(self.callback), inputs, states, 2
        )


if __name__ == "__main__":
    unittest.main()
