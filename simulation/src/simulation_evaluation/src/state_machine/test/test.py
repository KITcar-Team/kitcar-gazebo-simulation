"""Base Class for unit tests."""

import unittest


__copyright__ = "KITcar"


class Test(unittest.TestCase):
    """Base Class for unit tests."""

    def setUp(self):
        """Set up befor unit test."""
        self.callback_called = 0

    def callback(self):
        """Count callback."""
        self.callback_called += 1

    def state_machine_assert_on_input(self, state_machine, inputs, states, called):
        """Assert that state_machine is in the right state after inputing an input from the lists states and inputs."""
        self.setUp()

        for input, state in zip(inputs, states):
            state_machine.run(input)
            self.assertIs(state_machine.state, state)

        self.assertEqual(self.callback_called, called)
