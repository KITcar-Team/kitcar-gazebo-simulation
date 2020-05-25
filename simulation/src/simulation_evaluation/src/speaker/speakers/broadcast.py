from simulation.src.simulation_evaluation.src.speaker.speakers.speaker import Speaker

from simulation_evaluation.msg import Broadcast as BroadcastMsg

from typing import List

from . import export


@export
class BroadcastSpeaker(Speaker):
    """Keep high level information about the drive \
            (like speed, distance driven, current section).

    Instead of returning Speaker msgs this speaker returns a Broadcast msg.
    """

    def speak(self) -> List[BroadcastMsg]:
        """Return a new broadcast msg in a list."""
        msg = BroadcastMsg()

        msg.speed = self.car_speed
        msg.distance = self.arc_length

        msg.current_section = self.current_section

        return [msg]
