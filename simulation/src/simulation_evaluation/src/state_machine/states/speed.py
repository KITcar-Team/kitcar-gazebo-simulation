from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_evaluation.msg import State as StateMsg

from simulation.src.simulation_evaluation.src.state_machine.states.state import State

from ..state_machines.state_machine import StateMachine


class SpeedLimitIgnored(State):
    def __init__(self):
        super().__init__(
            description="The car ignored a speed limit.", value=StateMsg.SPEED_LIMIT_IGNORED
        )


class SpeedLimit(State):
    def __init__(self, limit):
        self.limit = limit
        super().__init__(
            description=f"The car is in a zone with speed limit {limit}.",
            value=getattr(StateMsg, f"SPEED_{limit}_ZONE"),
        )

        self.relevant_speed_limit = getattr(SpeakerMsg, f"SPEED_{limit-9}_{limit}")

    def next(self, state_machine: StateMachine, input_msg: int):
        if SpeakerMsg.SPEED_UNLIMITED_ZONE < input_msg <= SpeakerMsg.SPEED_90_ZONE:
            return getattr(state_machine, f"speed_{10 * (input_msg%10)}_limit")
        elif input_msg == 30:
            return state_machine.speed_no_limit
        elif self.relevant_speed_limit < input_msg and input_msg <= SpeakerMsg.SPEED_91_:
            return state_machine.failure_ignored_speed_limit

        return super().next(state_machine, input_msg)


class SpeedNoLimit(State):
    def __init__(self):
        super().__init__(
            description="The car is in a zone without a speed limit.",
            value=StateMsg.SPEED_UNLIMITED_ZONE,
        )

    def next(self, state_machine: StateMachine, input_msg: int):
        if SpeakerMsg.SPEED_UNLIMITED_ZONE < input_msg <= SpeakerMsg.SPEED_90_ZONE:
            return getattr(state_machine, f"speed_{10 * (input_msg%10)}_limit")

        return super().next(state_machine, input_msg)
