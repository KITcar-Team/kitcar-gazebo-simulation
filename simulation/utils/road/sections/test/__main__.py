"""Definition of the road section tests.

Whenever this module is executed, all of the tests included below are run.
"""

import road.sections.test.road_section_test as road_section
import road.sections.test.straight_road_test as straight_road
import road.sections.test.circular_test as circular
import road.sections.test.bezier_test as bezier
import road.sections.test.zebra_crossing_test as zebra_crossing
import road.sections.test.intersection_test as intersection
import road.sections.test.parking_test as parking

# Create test suite
import unittest
import sys

suite = unittest.TestSuite()


def append_test_cases(module):
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(module.ModuleTest))


append_test_cases(road_section)
append_test_cases(straight_road)
append_test_cases(circular)
append_test_cases(bezier)
append_test_cases(zebra_crossing)
append_test_cases(intersection)
append_test_cases(parking)

runner = unittest.TextTestRunner()
result = runner.run(suite)

# Ensure that failure in tests is recognized by CI
sys.exit(0 if result.wasSuccessful() else 1)
