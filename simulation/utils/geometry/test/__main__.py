"""Definition of the geometry module tests.

Whenever this module is executed, all of the tests included below are run."""

import simulation.utils.geometry.test.test_vector as vector
import simulation.utils.geometry.test.test_point as point
import simulation.utils.geometry.test.test_pose as pose
import simulation.utils.geometry.test.test_transform as transform
import simulation.utils.geometry.test.test_line_and_polygon as line_and_polygon

# Create test suite
import unittest
import sys

suite = unittest.TestSuite()


def append_test_cases(module):
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(module.ModuleTest))


append_test_cases(vector)
append_test_cases(point)
append_test_cases(pose)
append_test_cases(transform)
append_test_cases(line_and_polygon)

runner = unittest.TextTestRunner()
result = runner.run(suite)

sys.exit(0 if result.wasSuccessful() else 1)
