import math
import unittest
import functools
import random
from . import fake_msgs

from simulation.utils.geometry import Line, Vector, Pose, Polygon, Point

from simulation.src.simulation_evaluation.src.speaker.speakers import Speaker

import geometry_msgs.msg as geometry_msgs
import gazebo_simulation.msg as gazebo_sim_msgs


class ModuleTest(unittest.TestCase):
    def test_line_functions(self):
        """Create a line of points as middle line and check if the following functions/attributes work:

        * get_road_lines()
        * section_intervals
        * middle_line
        * arc_length
        * current_section

        """

        n_section = 10
        n_points = 100
        lane_width = 1

        # Create points
        lines = fake_msgs.create_points(
            section_count=n_section, point_count=n_points, offset_right=lane_width
        )

        # Fake section and lane msg proxies / usually in the groundtruth package
        section_msg_proxy = functools.partial(
            fake_msgs.section_msgs, section_count=n_section
        )
        lane_msg_proxy = functools.partial(fake_msgs.lane_msgs, lines)

        speaker = Speaker(section_proxy=section_msg_proxy, lane_proxy=lane_msg_proxy)

        test_line = Line()  # Keep track of the middle line of each section
        for idx, line_tuple in enumerate(lines):
            # Should return same left/middle/right line
            self.assertTupleEqual(line_tuple, speaker.get_road_lines(idx))

            # Test if speakers get interval works
            interval = speaker.section_intervals[idx]

            # Interval should start at end of previous section
            self.assertAlmostEqual(interval[0], test_line.length)

            test_line += line_tuple.middle

            self.assertAlmostEqual(interval[1], test_line.length)

            # Modify the pose
            speaker.car_pose = Pose(test_line.get_points()[-1], 0)  # last point of section
            # Test arc_length
            self.assertEqual(speaker.arc_length, test_line.length)

            speaker.car_pose = Pose(
                test_line.get_points()[-int(n_points / n_section / 2)], 0
            )  # Middle of section
            # Test current section

            self.assertEqual(speaker.current_section.id, idx)

        def assert_line_almost_eq(line1, line2):
            self.assertAlmostEqual(line1.get_points()[0], line2.get_points()[0])
            self.assertAlmostEqual(line1.get_points()[-1], line2.get_points()[-1])
            self.assertAlmostEqual(
                Polygon(line1, line2).area / line1.length / line2.length, 0, delta=1e-3
            )  # The polygon should have almost no area if the lines are approx. equal

        assert_line_almost_eq(
            speaker.middle_line, test_line
        )  # The polygon should have almost no area if the lines are approx. equal
        assert_line_almost_eq(
            speaker.right_line, test_line.parallel_offset(lane_width, side="right")
        )
        assert_line_almost_eq(
            speaker.left_line, test_line.parallel_offset(lane_width, side="left")
        )

    def test_listen_func(self):
        """Test if car state msg is processed correctly.

        * listen()
        """
        # Dummy msg
        pose = Pose(Point(1, 3), math.pi / 3)
        frame = Polygon([Point(0, -0.3), Point(0, 0.3), Point(0.5, 0.3), Point(0.5, -0.3)])
        linear_twist = Vector(2, 1, 3)

        car_msg = gazebo_sim_msgs.CarState()
        car_msg.pose = pose.to_geometry_msg()
        car_msg.frame = frame.to_geometry_msg()
        car_msg.twist = geometry_msgs.Twist()
        car_msg.twist.linear = linear_twist.to_geometry_msg()

        speaker = Speaker(
            section_proxy=fake_msgs.section_msgs, lane_proxy=fake_msgs.lane_msgs
        )

        speaker.listen(car_msg)

        # Test return values
        self.assertEqual(speaker.car_frame, frame)
        self.assertEqual(speaker.car_pose, pose)
        self.assertEqual(speaker.car_speed, abs(linear_twist))

    def test_overlapping_inside_funcs(self):
        """ Test polygon functions:

        * get_interval_for_polygon()
        * car_is_inside()
        * car_overlaps_with()
        * wheel_count_inside()
        """

        # Dummy msg
        pose = Pose(Point(1, 3), math.pi / 3)
        frame = Polygon(
            [Point(0.1, -0.3), Point(0.1, 0.3), Point(0.5, 0.3), Point(0.5, -0.3)]
        )

        # Fake section and lane msg proxies / usually in the groundtruth package
        section_msg_proxy = functools.partial(fake_msgs.section_msgs, section_count=2)
        lane_msg_proxy = functools.partial(
            fake_msgs.lane_msgs,
            fake_msgs.create_points(
                section_count=2, point_dist=0.1, point_count=40, deviation=0
            ),
        )

        speaker = Speaker(section_proxy=section_msg_proxy, lane_proxy=lane_msg_proxy)
        speaker.car_pose = pose
        speaker.car_frame = frame

        polygon1 = Polygon(
            [Point(0, -0.8), Point(0, 0.8), Point(0.8, 0.8), Point(0.8, -0.8)]
        )  # Contains frame completely
        polygon2 = Polygon(
            [Point(0, 0), Point(0, 0.8), Point(0.8, 0.8), Point(0.8, 0)]
        )  # Intersects with frame
        # Intersects with frame
        polygon3 = Polygon([Point(0, 0), Point(0, -0.8), Point(0.8, -0.8), Point(0.8, 0)])
        # p2 and p3 make up p1 together
        polygon4 = Polygon([Point(1, 1), Point(1, 2), Point(3, 2)])  # Disjoint to frame

        def assertTupleAlmostEqual(tuple1, tuple2):
            for t1, t2 in zip(tuple1, tuple2):
                self.assertAlmostEqual(t1, t2)

        assertTupleAlmostEqual(speaker.get_interval_for_polygon(polygon1), (0, 0.8))
        self.assertTrue(speaker.car_is_inside(polygon1))
        self.assertTrue(speaker.car_overlaps_with(polygon1))
        self.assertEqual(speaker.wheel_count_inside(polygon1), 4)

        assertTupleAlmostEqual(speaker.get_interval_for_polygon(polygon2), (0, 0.8))
        self.assertFalse(speaker.car_is_inside(polygon2))
        self.assertTrue(speaker.car_overlaps_with(polygon2))
        self.assertEqual(speaker.wheel_count_inside(polygon2), 2)

        assertTupleAlmostEqual(speaker.get_interval_for_polygon(polygon3), (0, 0.8))
        self.assertFalse(speaker.car_is_inside(polygon3))
        self.assertTrue(speaker.car_overlaps_with(polygon3))
        self.assertEqual(speaker.wheel_count_inside(polygon3), 2)

        assertTupleAlmostEqual(
            speaker.get_interval_for_polygon(polygon2, polygon3), (0, 0.8)
        )
        self.assertTrue(speaker.car_is_inside(polygon2, polygon3))
        self.assertTrue(speaker.car_overlaps_with(polygon2, polygon3))
        self.assertEqual(speaker.wheel_count_inside(polygon2, polygon3), 2)
        self.assertEqual(speaker.wheel_count_inside(polygon2, polygon3, in_total=True), 4)

        assertTupleAlmostEqual(speaker.get_interval_for_polygon(polygon4), (1, 3))
        self.assertFalse(speaker.car_is_inside(polygon4))
        self.assertFalse(speaker.car_overlaps_with(polygon4))
        self.assertEqual(speaker.wheel_count_inside(polygon4), 0)


if __name__ == "__main__":
    random.seed("KITCAR")
    unittest.main()
