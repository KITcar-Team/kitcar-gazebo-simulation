import itertools
import math
import unittest

import hypothesis.strategies as st
from hypothesis import given

import simulation.utils.road.sections.type as road_section_type
from simulation.utils.geometry import Line, Point, Transform, Vector
from simulation.utils.road.config import Config
from simulation.utils.road.sections.intersection import Intersection


class ModuleTest(unittest.TestCase):
    def test_init_combinations(self):
        @given(
            rule=st.sampled_from(
                [
                    Intersection.EQUAL,
                    Intersection.PRIORITY_STOP,
                    Intersection.PRIORITY_YIELD,
                    Intersection.YIELD,
                    Intersection.STOP,
                ]
            ),
            turn=st.sampled_from(
                [Intersection.LEFT, Intersection.STRAIGHT, Intersection.RIGHT]
            ),
            closing=st.sampled_from(
                [Intersection.LEFT, Intersection.STRAIGHT, Intersection.RIGHT]
            ),
            angle=st.floats(min_value=math.radians(70), max_value=math.radians(110)),
            size=st.floats(min_value=0.8, max_value=2),
            tf_x=st.floats(allow_nan=False),
            tf_y=st.floats(allow_nan=False),
            tf_rot=st.floats(min_value=0, max_value=2 * math.pi),
        )
        def combine(rule, turn, closing, angle, size, tf_x, tf_y, tf_rot):
            # print(rule, turn, closing, angle, size, tf_x, tf_y, tf_rot)
            if (
                turn == closing
                or size / 2 * math.sin(angle) <= math.sqrt(2) * Config.road_width
            ):
                return
            inter = Intersection(rule=rule, turn=turn, angle=angle, size=size)
            inter.set_transform(Transform(Point(tf_x, tf_y), tf_rot))
            for obj in itertools.chain(
                inter.traffic_signs, inter.surface_markings, inter.obstacles
            ):
                obj.frame

        combine()

    def test_intersection(self):
        TF = Transform([1, 1], math.pi / 2)
        TF = Transform([0, 0], 0)
        ANGLE = math.pi / 2
        CLOSING = None
        SIZE = 2
        RULE = Intersection.EQUAL

        inter = Intersection(angle=ANGLE, size=SIZE, rule=RULE, closing=CLOSING)
        inter.set_transform(TF)
        self.assertEqual(inter.__class__.TYPE, road_section_type.INTERSECTION)

        left_line_hz = TF * Line(
            [Point(0, Config.road_width), Point(SIZE, Config.road_width)]
        )
        middle_line_hz = TF * Line([Point(0, 0), Point(SIZE, 0)])
        right_line_hz = TF * Line(
            [Point(0, -Config.road_width), Point(SIZE, -Config.road_width)]
        )
        # vertical lines
        left_line_vt = TF * Line(
            [
                Point(SIZE / 2 - Config.road_width, SIZE / 2),
                Point(SIZE / 2 - Config.road_width, -SIZE / 2),
            ]
        )
        middle_line_vt = TF * Line([Point(SIZE / 2, SIZE / 2), Point(SIZE / 2, -SIZE / 2)])
        right_line_vt = TF * Line(
            [
                Point(SIZE / 2 + Config.road_width, SIZE / 2),
                Point(SIZE / 2 + Config.road_width, -SIZE / 2),
            ]
        )
        # rotate vt lines 110 - 90 degree
        angle = ANGLE - math.pi / 2
        inter_mid_point = TF * Point(1, 0)

        left_line_vt_tf = self.rotate_around_point(left_line_vt, angle, inter_mid_point)
        middle_line_vt_tf = self.rotate_around_point(middle_line_vt, angle, inter_mid_point)
        right_line_vt_tf = self.rotate_around_point(right_line_vt, angle, inter_mid_point)

        # intersection
        end_point_left_south = left_line_hz.intersection(left_line_vt_tf)
        end_point_middle_south = middle_line_hz.intersection(left_line_vt_tf)
        end_point_right_south = right_line_hz.intersection(left_line_vt_tf)
        start_point_mid_west = left_line_hz.intersection(middle_line_vt_tf)
        start_point_left_north = left_line_hz.intersection(right_line_vt_tf)
        start_point_mid_north = middle_line_hz.intersection(right_line_vt_tf)
        start_point_right_north = right_line_hz.intersection(right_line_vt_tf)
        start_point_mid_east = right_line_hz.intersection(middle_line_vt_tf)

        self.assert_lines_almost_equal(
            inter.left_line_south,
            Line([left_line_hz.get_points()[0], end_point_left_south]),
        )
        self.assert_lines_almost_equal(
            inter.middle_line_south,
            Line([middle_line_hz.get_points()[0], end_point_middle_south]),
        )
        self.assert_lines_almost_equal(
            inter.right_line_south,
            Line([right_line_hz.get_points()[0], end_point_right_south]),
        )

        self.assert_lines_almost_equal(
            inter.left_line_west,
            Line([end_point_left_south, left_line_vt_tf.get_points()[0]]),
        )
        self.assert_lines_almost_equal(
            inter.middle_line_west,
            Line([start_point_mid_west, middle_line_vt_tf.get_points()[0]]),
        )
        self.assert_lines_almost_equal(
            inter.right_line_west,
            Line([start_point_left_north, right_line_vt_tf.get_points()[0]]),
        )

        self.assert_lines_almost_equal(
            inter.left_line_north,
            Line([start_point_left_north, left_line_hz.get_points()[1]]),
        )
        self.assert_lines_almost_equal(
            inter.middle_line_north,
            Line([start_point_mid_north, middle_line_hz.get_points()[1]]),
        )
        self.assert_lines_almost_equal(
            inter.right_line_north,
            Line([start_point_right_north, right_line_hz.get_points()[1]]),
        )

        self.assert_lines_almost_equal(
            inter.left_line_east,
            Line([start_point_right_north, right_line_vt_tf.get_points()[1]]),
        )
        self.assert_lines_almost_equal(
            inter.middle_line_east,
            Line([start_point_mid_east, middle_line_vt_tf.get_points()[1]]),
        )
        self.assert_lines_almost_equal(
            inter.right_line_east,
            Line([end_point_right_south, left_line_vt_tf.get_points()[1]]),
        )

    # rotate around middle point of intersection
    def rotate_around_point(self, rotate_this, angle, rotate_around):
        add_mid = Transform(Vector(rotate_around), 0)
        subtract_mid = Transform(-1 * Vector(rotate_around), 0)
        rot = Transform(Point(0, 0), angle)
        rotated = add_mid * (rot * (subtract_mid * rotate_this))
        return rotated

    ###
    # Helper functions
    def assertPointAlmostEqual(self, p1, p2):
        self.assertAlmostEqual(p1.x, p2.x)
        self.assertAlmostEqual(p1.y, p2.y)
        self.assertAlmostEqual(p1.z, p2.z)

    def assert_lines_almost_equal(self, line1, line2):
        for p1, p2 in zip(line1.get_points(), line2.get_points()):
            self.assertPointAlmostEqual(p1, p2)


if __name__ == "__main__":
    unittest.main()
