"""Definition of the state_machine module tests.

Whenever this module is executed, all of the tests included below are run.
"""

import simulation.src.simulation_evaluation.src.state_machine.test.test_overtaking_state_machine as overtaking
import simulation.src.simulation_evaluation.src.state_machine.test.test_parking_state_machine as parking
import simulation.src.simulation_evaluation.src.state_machine.test.test_priority_state_machine as priority
import simulation.src.simulation_evaluation.src.state_machine.test.test_progress_state_machine as progress

# Create test suite
import unittest

suite = unittest.TestSuite()


def append_test_cases(module):
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(module.ModuleTest))


append_test_cases(overtaking)
append_test_cases(parking)
append_test_cases(priority)
append_test_cases(progress)

runner = unittest.TextTestRunner()
result = runner.run(suite)

if len(result.errors) > 0:
    exit(1)
