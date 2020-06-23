# -*- coding: utf-8 -*-

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation.src.simulation_evaluation.src.state_machine.state_machines.overtaking import (
    OvertakingStateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.state_machines.parking import (
    ParkingStateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.state_machines.priority import (
    PriorityStateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.state_machines.progress import (
    ProgressStateMachine,
)

__copyright__ = "KITcar"


progress = ProgressStateMachine(None)
overtaking = OvertakingStateMachine(None)
parking = ParkingStateMachine(None)
priority = PriorityStateMachine(None)

directory = "content/simulation_evaluation/graphs/"

progress.generate_graph(SpeakerMsg, directory=directory, filename="progress")
overtaking.generate_graph(SpeakerMsg, directory=directory, filename="overtaking")
parking.generate_graph(SpeakerMsg, directory=directory, filename="parking")
priority.generate_graph(SpeakerMsg, directory=directory, filename="priority")
