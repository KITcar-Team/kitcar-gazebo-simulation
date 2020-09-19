"""Definition of the state_machine module tests.

Whenever this module is executed, all of the tests included below are run.
"""

# Create test suite
import unittest

from . import test_lane_state_machine as lane
from . import test_overtaking_state_machine as overtaking
from . import test_parking_state_machine as parking
from . import test_priority_state_machine as priority
from . import test_progress_state_machine as progress

suite = unittest.TestSuite()


def append_test_cases(module):
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(module.ModuleTest))


append_test_cases(lane)
append_test_cases(overtaking)
append_test_cases(parking)
append_test_cases(priority)
append_test_cases(progress)

runner = unittest.TextTestRunner()
result = runner.run(suite)

if len(result.errors) > 0:
    exit(1)
