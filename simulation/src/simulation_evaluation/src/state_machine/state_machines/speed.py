from typing import Callable

from simulation.src.simulation_evaluation.src.state_machine.states.speed import (
    SpeedLimit,
    SpeedLimitIgnored,
    SpeedNoLimit,
)

from .state_machine import StateMachine


class SpeedStateMachine(StateMachine):
    speed_no_limit: "State" = SpeedNoLimit()  # noqa: F821
    """"""
    failure_ignored_speed_limit: "State" = SpeedLimitIgnored()  # noqa: F821

    def __init__(self, callback: Callable[[], None]):
        super(SpeedStateMachine, self).__init__(
            state_machine=self.__class__,
            initial_state=SpeedStateMachine.speed_no_limit,
            callback=callback,
        )


for limit in range(1, 10):
    setattr(SpeedStateMachine, f"speed_{limit * 10}_limit", SpeedLimit(limit * 10))
