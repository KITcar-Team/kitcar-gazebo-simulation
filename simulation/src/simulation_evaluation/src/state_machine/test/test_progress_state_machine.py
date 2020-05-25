"""ProgressStateMachine ModuleTest."""

import unittest

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation.src.simulation_evaluation.src.state_machine.state_machines.progress import (
    ProgressStateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.test.test import Test


__copyright__ = "KITcar"


class ModuleTest(Test):
    """Test if the ProgressStateMachine behaves as expected."""

    def test_complete_drive(self):
        """Test full drive."""
        prog_sm = ProgressStateMachine(self.callback)

        # Should be in start state
        self.assertIs(prog_sm.state, ProgressStateMachine.before_start)

        # Input some unrelated msg
        prog_sm.run(SpeakerMsg.COLLISION)  # Should not affect statemachine!!
        self.assertIs(prog_sm.state, ProgressStateMachine.before_start)

        self.assertEqual(self.callback_called, 0)

        # Input driving msg
        prog_sm.run(SpeakerMsg.DRIVING_ZONE)
        self.assertIs(prog_sm.state, ProgressStateMachine.running)

        self.assertEqual(self.callback_called, 1)

        # Input finished msg
        prog_sm.run(SpeakerMsg.END_ZONE)
        self.assertIs(prog_sm.state, ProgressStateMachine.finished)

        self.assertEqual(self.callback_called, 2)


if __name__ == "__main__":
    unittest.main()
