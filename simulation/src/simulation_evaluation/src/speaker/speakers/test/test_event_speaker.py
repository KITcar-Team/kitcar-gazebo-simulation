import unittest
import random
import functools
from simulation.utils.geometry import Line, Polygon, Transform

from simulation.src.simulation_evaluation.src.speaker.speakers import EventSpeaker

from simulation_evaluation.msg import Speaker as SpeakerMsg

from . import fake_msgs
from . import utils


class ModuleTest(unittest.TestCase):
    def test_event_speaker(self):
        """Test properties and speak function of event speaker."""

        # Obstacles
        o = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        obstacles = [
            o,
            Transform([2, 0], 0) * o,
            Transform([4, 0], 0) * o,
            Transform([5, 0], 0) * o,
            Transform([7, 0], 0) * o,
        ]

        # parking
        p = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        left_spots = [
            p,
            Transform([1, 0], 0) * p,
            Transform([2, 0], 0) * p,
            Transform([3, 0], 0) * p,
            Transform([4, 0], 0) * p,
        ]
        right_spots = [
            Transform([0, 1], 0) * p,
            Transform([1, 1], 0) * p,
            Transform([2, 1], 0) * p,
            Transform([3, 1], 0) * p,
            Transform([4, 1], 0) * p,
        ]

        # Fake section and lane msg proxies / usually in the groundtruth package
        section_msg_proxy = functools.partial(fake_msgs.section_msgs, section_count=1)
        lane_msg_proxy = functools.partial(fake_msgs.lane_msgs, [])
        obstacle_msg_proxy = functools.partial(fake_msgs.obstacle_msgs, obstacles)
        parking_msg_proxy = functools.partial(
            fake_msgs.parking_msgs, right_spots, left_spots, Line(), Line()
        )

        speaker = EventSpeaker(
            section_proxy=section_msg_proxy,
            lane_proxy=lane_msg_proxy,
            obstacle_proxy=obstacle_msg_proxy,
            parking_proxy=parking_msg_proxy,
            parking_spot_buffer=0.0,
            min_parking_wheels=3,
        )

        self.assertTrue(utils.polygon_list_almost_equal(speaker.obstacles, obstacles))
        self.assertTrue(
            utils.polygon_list_almost_equal(speaker.parking_spots, left_spots + right_spots)
        )

        # Put vehicle inside of a polygon
        veh = Polygon([[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]])

        # with buffer for parking spots
        speaker = EventSpeaker(
            section_proxy=section_msg_proxy,
            lane_proxy=lane_msg_proxy,
            obstacle_proxy=obstacle_msg_proxy,
            parking_proxy=parking_msg_proxy,
            parking_spot_buffer=0.01,
            min_parking_wheels=3,
        )

        def tf(x, y):
            return Transform([x, y], 0) * veh

        self.assertTrue(utils.assert_msgs_for_pos(speaker, tf(5, 5)))
        self.assertTrue(utils.assert_msgs_for_pos(speaker, tf(2, 0), SpeakerMsg.COLLISION))
        self.assertTrue(
            utils.assert_msgs_for_pos(
                speaker, tf(2.5, 0.5), SpeakerMsg.COLLISION, SpeakerMsg.PARKING_SPOT
            )
        )
        self.assertTrue(
            utils.assert_msgs_for_pos(speaker, tf(2.5, 1.5), SpeakerMsg.PARKING_SPOT)
        )
        self.assertTrue(
            utils.assert_msgs_for_pos(speaker, tf(2.5, 1.5), SpeakerMsg.PARKING_SPOT)
        )
        self.assertTrue(
            utils.assert_msgs_for_pos(
                speaker, tf(0.3, 0.5), SpeakerMsg.PARKING_SPOT, SpeakerMsg.COLLISION
            )
        )


if __name__ == "__main__":
    random.seed("KITCAR")
    unittest.main()
