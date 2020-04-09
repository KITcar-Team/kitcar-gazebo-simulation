# -*- coding: utf-8 -*-
"""Base class State."""

__copyright__ = "KITcar"


class State:
    """Base class State used for StateMachine."""

    def __init__(self, description: str, value: int):
        """Initialize State.

        Arguments:
            description: Human readable description of this state
            value: Value of State Message (Defined in msg/State.msg)
        """
        self.description = description
        self.value = value

    def next(self, state_machine: "StateMachine", input_msg: int):  # noqa: F821
        """Next state.

        Arguments:
            state_machine (StateMachine): On which state machine the states gets executed
            input_msg: Integer of message

        Returns:
            self (--> state can no longer change)
        """
        return self
