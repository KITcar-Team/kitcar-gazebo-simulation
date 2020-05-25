"""Definition of the speaker module tests.

Whenever this module is executed, all of the tests included below are run."""

import simulation.src.simulation_evaluation.src.speaker.speakers.test.test_speaker as speaker
import simulation.src.simulation_evaluation.src.speaker.speakers.test.test_event_speaker as event
import simulation.src.simulation_evaluation.src.speaker.speakers.test.test_area_speaker as area
import simulation.src.simulation_evaluation.src.speaker.speakers.test.test_speed_speaker as speed
import simulation.src.simulation_evaluation.src.speaker.speakers.test.test_zone_speaker as zone

# Create test suite
import unittest

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

if len(result.errors) > 0:
    exit(1)
