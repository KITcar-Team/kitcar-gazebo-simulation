"""PriorityStateMachine ModuleTest."""

import unittest

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation.src.simulation_evaluation.src.state_machine.state_machines.priority import (
    PriorityStateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.test.test import Test


__copyright__ = "KITcar"


class ModuleTest(Test):
    """Test if the PriorityStateMachine behaves as expected."""

    def test_correct_behavior(self):
        """Test if correct behavior works as expected."""
        inputs = [
            SpeakerMsg.STOP_ZONE,
            SpeakerMsg.SPEED_STOPPED,
            SpeakerMsg.NO_STOP_ZONE,
            SpeakerMsg.HALT_ZONE,
            SpeakerMsg.SPEED_HALTED,
        ]
        states = [
            PriorityStateMachine.in_stop_zone,
            PriorityStateMachine.successfully_stopped,
            PriorityStateMachine.off,
            PriorityStateMachine.in_halt_zone,
            PriorityStateMachine.off,
        ]

        self.state_machine_assert_on_input(
            PriorityStateMachine(self.callback), inputs, states, 5
        )

    def test_input_multiple_times(self):
        """Test if inputting speaker msgs multiple times changes anything (which it shouldn't)."""
        inputs = [
            SpeakerMsg.STOP_ZONE,
            SpeakerMsg.STOP_ZONE,
            SpeakerMsg.SPEED_STOPPED,
            SpeakerMsg.SPEED_STOPPED,
            SpeakerMsg.NO_STOP_ZONE,
            SpeakerMsg.NO_STOP_ZONE,
            SpeakerMsg.HALT_ZONE,
            SpeakerMsg.HALT_ZONE,
            SpeakerMsg.SPEED_HALTED,
            SpeakerMsg.SPEED_HALTED,
        ]
        states = [
            PriorityStateMachine.in_stop_zone,
            PriorityStateMachine.in_stop_zone,
            PriorityStateMachine.successfully_stopped,
            PriorityStateMachine.successfully_stopped,
            PriorityStateMachine.off,
            PriorityStateMachine.off,
            PriorityStateMachine.in_halt_zone,
            PriorityStateMachine.in_halt_zone,
            PriorityStateMachine.off,
            PriorityStateMachine.off,
        ]

        self.state_machine_assert_on_input(
            PriorityStateMachine(self.callback), inputs, states, 5
        )

    def test_unrelated_msgs(self):
        """Test if inputting unrelated speaker msgs changes anything (which it shouldn't)."""
        inputs = [
            SpeakerMsg.STOP_ZONE,
            SpeakerMsg.SPEED_0,
            SpeakerMsg.SPEED_STOPPED,
            SpeakerMsg.DRIVING_ZONE,
            SpeakerMsg.NO_STOP_ZONE,
            SpeakerMsg.RIGHT_LANE,
            SpeakerMsg.HALT_ZONE,
            SpeakerMsg.DRIVING_ZONE,
            SpeakerMsg.SPEED_HALTED,
            SpeakerMsg.SPEED_0,
        ]
        states = [
            PriorityStateMachine.in_stop_zone,
            PriorityStateMachine.in_stop_zone,
            PriorityStateMachine.successfully_stopped,
            PriorityStateMachine.successfully_stopped,
            PriorityStateMachine.off,
            PriorityStateMachine.off,
            PriorityStateMachine.in_halt_zone,
            PriorityStateMachine.in_halt_zone,
            PriorityStateMachine.off,
            PriorityStateMachine.off,
        ]

        self.state_machine_assert_on_input(
            PriorityStateMachine(self.callback), inputs, states, 5
        )


if __name__ == "__main__":
    unittest.main()
