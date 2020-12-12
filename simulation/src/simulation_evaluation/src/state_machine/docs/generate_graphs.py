from simulation_evaluation.msg import Speaker as SpeakerMsg

from ..state_machines.lane import LaneStateMachine
from ..state_machines.overtaking import OvertakingStateMachine
from ..state_machines.parking import ParkingStateMachine
from ..state_machines.priority import PriorityStateMachine
from ..state_machines.progress import ProgressStateMachine
from ..state_machines.speed import SpeedStateMachine

lane = LaneStateMachine(None)
progress = ProgressStateMachine(None)
overtaking = OvertakingStateMachine(None)
parking = ParkingStateMachine(None)
priority = PriorityStateMachine(None)
speed = SpeedStateMachine(None)

directory = "content/simulation_evaluation/graphs/"

lane.generate_graph(SpeakerMsg, directory=directory, filename="lane")
progress.generate_graph(SpeakerMsg, directory=directory, filename="progress")
overtaking.generate_graph(SpeakerMsg, directory=directory, filename="overtaking")
parking.generate_graph(SpeakerMsg, directory=directory, filename="parking")
priority.generate_graph(SpeakerMsg, directory=directory, filename="priority")
speed.generate_graph(SpeakerMsg, directory=directory, filename="speed")
