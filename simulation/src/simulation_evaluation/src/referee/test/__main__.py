"""Definition of the referee module tests.

Whenever this module is executed, all of the tests included below are run.
"""

import simulation.src.simulation_evaluation.src.referee.test.test_referee as referee

# Create test suite
import unittest

suite = unittest.TestSuite()

suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(referee.ModuleTest))

runner = unittest.TextTestRunner()
result = runner.run(suite)

if len(result.errors) > 0:
    exit(1)
