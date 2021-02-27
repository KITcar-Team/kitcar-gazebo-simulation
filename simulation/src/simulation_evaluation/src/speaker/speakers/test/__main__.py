"""Definition of the speaker module tests.

Whenever this module is executed, all of the tests included below are run.

Run these tests with::

    python3 -m simulation.src.simulation_evaluation.src.speaker.speakers.test
"""

# Create test suite
import sys
import unittest

from . import test_area_speaker as area
from . import test_event_speaker as event
from . import test_speaker as speaker
from . import test_speed_speaker as speed
from . import test_zone_speaker as zone

suite = unittest.TestSuite()


def append_test_cases(module):
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(module.ModuleTest))


append_test_cases(speaker)
append_test_cases(event)
append_test_cases(area)
append_test_cases(speed)
append_test_cases(zone)

runner = unittest.TextTestRunner()
result = runner.run(suite)

sys.exit(0 if result.wasSuccessful() else 1)
