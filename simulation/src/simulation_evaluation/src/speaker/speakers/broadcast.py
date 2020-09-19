from typing import List

from simulation_evaluation.msg import Broadcast as BroadcastMsg

from .speaker import Speaker


class BroadcastSpeaker(Speaker):
    """Keep high level information about the drive (e.g. speed, distance driven).

    Instead of returning Speaker msgs this speaker returns a Broadcast msg.
    """

    def speak(self) -> List[BroadcastMsg]:
        """Return a new broadcast msg in a list."""
        msg = BroadcastMsg()

        msg.speed = self.car_speed
        msg.distance = self.arc_length

        msg.current_section = self.current_section

        return [msg]
