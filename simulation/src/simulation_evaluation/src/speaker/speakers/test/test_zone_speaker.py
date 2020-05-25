import unittest

from simulation_evaluation.msg import Speaker as SpeakerMsg

import random
import numpy

from simulation.utils.geometry import Line, Pose, Polygon, Transform
import simulation.utils.road.sections.type as road_section_type

from simulation.src.simulation_evaluation.src.speaker.speakers import ZoneSpeaker

import functools

import simulation_groundtruth.msg as groundtruth_msgs
import simulation_groundtruth.srv as groundtruth_srvs

from typing import List

from . import fake_msgs


class ModuleTest(unittest.TestCase):
    def setUp(self):

        n_section = 5
        section_types = [
            road_section_type.LEFT_CIRCULAR_ARC,
            road_section_type.CUBIC_BEZIER,
            road_section_type.INTERSECTION,
            road_section_type.INTERSECTION,
            road_section_type.PARKING_AREA,
        ]
        n_points = 100
        lane_width = 0.4
        overtaking_buffer = 2
        self.yield_distance = (-1, -0.5)

        # Create points
        lines = fake_msgs.create_points(
            section_count=n_section,
            point_count=n_points,
            offset_right=lane_width,
            direction=0,
            deviation=0,
        )

        # Fake section and lane msg proxies / usually in the groundtruth package
        section_msg_proxy = functools.partial(
            fake_msgs.section_msgs, section_count=n_section, section_types=section_types
        )
        lane_msg_proxy = functools.partial(fake_msgs.lane_msgs, lines)
        parking_msg_proxy = functools.partial(fake_msgs.parking_msgs, [], [], [], [])

        # Create obstacles
        self.overtaking_intervals = []
        self.inv_overtaking_intervals = []
        obstacles = []
        o = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])

        self.inv_overtaking_intervals.append((0, 3))

        width = 1
        prev_x = 0

        def next_obstacles(*xs):
            nonlocal prev_x
            for x in xs:
                obstacles.append(Transform([x, 0], 0) * o)
            self.inv_overtaking_intervals.append((prev_x, xs[0] - overtaking_buffer))
            self.overtaking_intervals.append(
                (xs[0] - overtaking_buffer, xs[-1] + width + overtaking_buffer)
            )
            prev_x = xs[-1] + width + overtaking_buffer

        # Create obstacles with overlapping and non overlapping buffers
        next_obstacles(3)
        next_obstacles(10)
        next_obstacles(16, 18)
        next_obstacles(23, 24, 26)

        self.inv_overtaking_intervals.append((prev_x, 100))

        def obstacle_proxy(id):
            if id == 0:
                return fake_msgs.obstacle_msgs(obstacles[0:4], id)
            if id == 1:
                return fake_msgs.obstacle_msgs(obstacles[4:], id)
            return fake_msgs.obstacle_msgs([], id)

        # Create intersection proxy
        self._rules = {
            2: groundtruth_srvs.IntersectionSrvResponse.YIELD,
            3: groundtruth_srvs.IntersectionSrvResponse.STOP,
        }

        def intersection_proxy(id):
            lane = groundtruth_msgs.Lane()
            if id == 2:
                lane.middle_line = Line([[19, 0], [20, 0]]).to_geometry_msgs()
            if id == 3:
                lane.middle_line = Line([[29, 0], [30, 0]]).to_geometry_msgs()
            return fake_msgs.intersection_msg(rule=self._rules[id], south=lane)

        self.speaker = ZoneSpeaker(
            section_proxy=section_msg_proxy,
            lane_proxy=lane_msg_proxy,
            parking_proxy=parking_msg_proxy,
            obstacle_proxy=obstacle_proxy,
            intersection_proxy=intersection_proxy,
            overtaking_buffer=overtaking_buffer,
            start_zone_buffer=0.5,
            end_zone_buffer=0.5,
            yield_distance=self.yield_distance,
        )

    def test_zones(self):
        self.assertListEqual(self.speaker.overtaking_zones, self.overtaking_intervals)

    def test_stop_zones(self):
        self._rules[2] = groundtruth_srvs.IntersectionSrvResponse.STOP
        self._rules[3] = groundtruth_srvs.IntersectionSrvResponse.STOP
        self.assertSetEqual(set(self.speaker.stop_zones), {(19, 19.5), (29, 29.5)})
        self.assertSetEqual(set(self.speaker.halt_zones), set())

    def test_halt_zones(self):
        self._rules[2] = groundtruth_srvs.IntersectionSrvResponse.YIELD
        self._rules[3] = groundtruth_srvs.IntersectionSrvResponse.YIELD
        self.assertSetEqual(set(self.speaker.stop_zones), set())
        self.assertSetEqual(set(self.speaker.halt_zones), {(19, 19.5), (29, 29.5)})

    def test_speak_function(self):
        """Test the speakers msgs by putting the car somewhere and checking the output."""

        msg_options = dict()

        msg_options[SpeakerMsg.START_ZONE] = {(0, 0.5)}
        msg_options[SpeakerMsg.DRIVING_ZONE] = {(0.5, 99.5)}
        msg_options[SpeakerMsg.END_ZONE] = {(99.5, 100)}

        msg_options[SpeakerMsg.OVERTAKING_ZONE] = self.overtaking_intervals
        msg_options[SpeakerMsg.NO_OVERTAKING_ZONE] = self.inv_overtaking_intervals

        msg_options[SpeakerMsg.PARKING_ZONE] = {(80, 100)}
        msg_options[SpeakerMsg.NO_PARKING_ZONE] = {(0, 80)}

        msg_options[SpeakerMsg.HALT_ZONE] = {
            (20 + self.yield_distance[0], 20 + self.yield_distance[1])
        }
        msg_options[SpeakerMsg.STOP_ZONE] = {
            (30 + self.yield_distance[0], 30 + self.yield_distance[1])
        }
        msg_options[SpeakerMsg.NO_STOP_ZONE] = {
            (0, 20 + self.yield_distance[0]),
            (20 + self.yield_distance[1], 30 + self.yield_distance[0]),
            (30 + self.yield_distance[1], 100),
        }

        msg_expects = {
            (SpeakerMsg.START_ZONE, SpeakerMsg.DRIVING_ZONE, SpeakerMsg.END_ZONE),
            (SpeakerMsg.OVERTAKING_ZONE, SpeakerMsg.NO_OVERTAKING_ZONE),
            (SpeakerMsg.PARKING_ZONE, SpeakerMsg.NO_PARKING_ZONE),
            (SpeakerMsg.HALT_ZONE, SpeakerMsg.STOP_ZONE, SpeakerMsg.NO_STOP_ZONE),
        }

        for x in numpy.arange(0, 100, 0.1):
            # create car msg
            self.speaker.car_pose = Pose([x, 0], 0)

            result: List[SpeakerMsg] = self.speaker.speak()
            msg_types = {msg.type for msg in result}

            # Check that each of the expected msgs is in what the speaker says
            for exp in msg_expects:
                self.assertTrue(len(set(exp) & msg_types) > 0)

            for msg in msg_types:
                opts = msg_options[msg]

                # Check if thats correct
                self.assertTrue(
                    any(interval[0] <= x and interval[1] >= x for interval in opts)
                )


if __name__ == "__main__":
    random.seed("KITCAR")
    unittest.main()
