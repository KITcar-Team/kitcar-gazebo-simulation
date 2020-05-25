# -*- coding: utf-8 -*-
"""Base class StateMachine."""

from typing import Callable, Dict, Type

from graphviz import Digraph

from simulation_evaluation.msg import State as StateMsg
from simulation.src.simulation_evaluation.src.state_machine.states.failure import (
    FailureCollision,
    FailureLeft,
    FailureOffRoad,
)
from simulation.src.simulation_evaluation.src.state_machine.states.state import State

__copyright__ = "KITcar"


class StateMachine:
    """Base class for all state machines.

    This class handels state changes for a StateMachine. It also adds the methods's info, value and msg which return \
        each values of the current state. Additionally, it gives a property which returns all states and a method \
        which creates a graph with all states.

    Attributes:
        state_machine (StateMachine): State machine used for all operations
        state (State): Current state
        callback (method): Function which gets executed when the state changes
    """

    failure_collision: State = FailureCollision()
    """End state when driving into an obstacle"""
    failure_off_road: State = FailureOffRoad()
    """End state when driving of the road"""
    failure_left: State = FailureLeft()
    """End state when driving onto the left lane"""

    def __init__(
        self,
        state_machine: "StateMachine",
        initial_state: State,
        callback: Callable[[], None],
    ):
        """Initialize StateMachine.

        Arguments:
            state_machine (StateMachine): The StateMachine with all states
            initial_state (State): State the StateMachine should start with
            callback: Function which gets executed when the state changes
        """
        self.state_machine = state_machine
        self.state = initial_state
        self._initial_state = initial_state
        self.callback = callback

    def callback_on_state_change(func):
        """Decorator which executes self.callback when the state of state machine has changed.

        Arguments:
            func: Function to wrap

        Returns:
            Result of func
        """

        def wrapper(self, *args, **kwargs):
            prev_state = self.state
            result = func(self, *args, **kwargs)
            if self.state != prev_state:
                self.callback()

            return result

        return wrapper

    @callback_on_state_change
    def run(self, input_msg: int):
        """Update self.state with new state and execute self.callback if state changes.

        Arguments:
            input_msg: Integer of message
        """
        self.state = self.state.next(self.state_machine, input_msg)

    def info(self) -> str:
        """Get human-readable description of current state.

        Returns:
            String of current description
        """
        return self.state.description

    def value(self) -> int:
        """Get value of current state.

        Returns:
            Integer of current value
        """
        return self.state.value

    def msg(self) -> StateMsg:
        """Get message of current state.

        Returns:
            StateMsg of current state
        """
        msg = StateMsg()
        msg.state = self.value()

        return msg

    @property
    def all_states(self) -> Dict[int, State]:
        """Property which gives all available states inherting from State in current StateMachine."""
        # Collect all base clases of StateMachine and parent
        class_hierarchie = list(self.__class__.__bases__)
        class_hierarchie.append(self.__class__)
        # Select all states from class atributes which are some child of State
        return {
            state.value: state
            for cl in class_hierarchie
            for state in cl.__dict__.values()
            if issubclass(state.__class__, State)
        }

    @callback_on_state_change
    def set(self, new_msg: StateMsg) -> bool:
        """Manually set state in StateMachine.

        Arguments:
            new_msg: StateMsg of next state to be set

        Returns:
            Boolean if state was successfully set
        """
        if new_msg.state in self.all_states:
            self.state = self.all_states[new_msg.state]
            return True

        return False

    def generate_graph(
        self,
        messages: Type[StateMsg],
        directory: str = "",
        filename: str = "graph",
        accent_color: str = "grey",
        shape: str = "oval",
        shape_failure: str = "rect",
        view: bool = False,
        save_to_file: bool = True,
    ) -> str:
        """Generate Graph for current StateMachine.

        Arguments:
            messages: Object of all messages
            directory: Directory where the output file should be saved
            filename: Name of output file
            accent_color: Accent color of graph
            shape: Default node shape
            shape_failure: Failure node shape
            view: If the graph should be shown to the user
            save_to_file: If the graph should be saved to a svg file

        Returns:
            A string of the generated source code of the graph

        .. note::
            You can find documentation on graphviz on there `homepage <https://graphviz.org/documentation/>`_ and on \
                `readthedocs <https://graphviz.readthedocs.io/en/stable/manual.html#basic-usage>`_.
        """
        # Get all message names and there ids defined in messages
        msgs = [
            (name, val)
            for name, val in messages.__dict__.items()
            if isinstance(val, int) and not name[0] == "_"
        ]

        # Setup Graph
        g = Digraph()
        g.attr("edge", fontsize="8")

        # Check all messages in all states and generate graph accordingly
        for _, state in self.all_states.items():
            for msg, index in msgs:
                parent = state.__class__.__name__

                # Add message as label to edge
                g.attr("edge", label=msg)

                try:
                    next_state = state.next(self.state_machine, index)

                    child = next_state.__class__.__name__

                    if next_state != state:
                        if child.startswith("Failure"):
                            # Add Styling for failure state
                            g.node(
                                child,
                                label=child,
                                shape=shape_failure,
                                color=accent_color,
                                fontsize="10",
                            )
                            g.edge(parent, child, color=accent_color)
                            continue

                        elif self._initial_state == next_state:
                            # Add Styling for initial state
                            # FIXME: A state that does not receive a message doesn't gets generate this way. Instead,
                            # he gets generated by creating an edge (of other node) -> Therefor he gets no styling
                            g.node(
                                child,
                                label=child,
                                shape=shape,
                                fillcolor=accent_color,
                                style="filled",
                            )
                            g.edge(parent, child)
                            continue

                        g.node(child, label=child, shape=shape)
                        g.edge(parent, child)

                # Is thrown in states used for ProgressStateMachine when something unexpected happened
                except AssertionError:
                    child = "AssertionError"

                    # Add Styling for assertion error "state"
                    g.node(
                        child,
                        label=child,
                        shape=shape_failure,
                        color=accent_color,
                        fontsize="10",
                    )
                    g.edge(parent, child, color=accent_color)

        if save_to_file:
            g.render(
                directory=directory,
                filename=filename,
                cleanup=True,
                format="svg",
                view=view,
            )

        return g.source
