import unittest
import math

import simulation.utils.road.sections.type as road_section_type

from simulation.utils.geometry import Transform, Point, Vector

from simulation.utils.road.sections import (
    CubicBezier,
    QuadBezier,
)


class ModuleTest(unittest.TestCase):
    def assert_equal_angle(self, angle1, angle2):
        self.assertAlmostEqual(
            # Use math.sin to compare angles
            math.sin(angle1),
            math.sin(angle2),
            delta=0.01,
        )

    def test_quad_bezier_curve(self):
        TF = Transform([1, 1], math.pi / 2)
        POINTS = [Point(5, 0), Point(2, 5)]

        # simple quad bezier
        qb = QuadBezier(p1=POINTS[0], p2=POINTS[1], transform=TF)
        self.assertEqual(qb.__class__.TYPE, road_section_type.QUAD_BEZIER)
        self.assertPointAlmostEqual(qb.middle_line.get_points()[0], TF * Point(0, 0))
        self.assertPointAlmostEqual(qb.middle_line.get_points()[-1], TF * POINTS[1])

        test_end_angle = (
            math.acos(
                (Vector(-3, 5) * Vector(1, 0)) / (abs(Vector(-3, 5)) * abs(Vector(1, 0)))
            )
            + TF.get_angle()
        )
        self.assert_equal_angle(qb.get_ending()[0].get_angle(), test_end_angle)

    def test_cubic_bezier_curve(self):
        TF = Transform([1, 1], math.pi / 2)
        POINTS = [Point(5, 0), Point(2, 5), Point(5, 5)]

        cb = CubicBezier(p1=POINTS[0], p2=POINTS[1], p3=POINTS[2], transform=TF)
        self.assertEqual(cb.__class__.TYPE, road_section_type.CUBIC_BEZIER)
        bezier_points = cb.middle_line.get_points()
        self.assertPointAlmostEqual(bezier_points[0], TF * Point(0, 0))
        self.assertPointAlmostEqual(bezier_points[-1], TF * POINTS[-1])
        self.assert_equal_angle(
            cb.get_beginning()[0].get_angle(), -math.pi + TF.get_angle()
        )
        self.assert_equal_angle(cb.get_ending()[0].get_angle(), TF.get_angle())

    ###
    # Helper functions
    def assertPointAlmostEqual(self, p1, p2):
        self.assertAlmostEqual(p1.x, p2.x)
        self.assertAlmostEqual(p1.y, p2.y)
        self.assertAlmostEqual(p1.z, p2.z)

    def assertTupleAlmostEqual(self, tuple1, tuple2, places=7):
        print(places)
        self.assertAlmostEqual(tuple1[0].x, tuple2[0].x, places=places)
        self.assertAlmostEqual(tuple1[0].y, tuple2[0].y, places=places)
        self.assertAlmostEqual(tuple1[1], tuple2[1], places=places)
        self.assertAlmostEqual(tuple1[2], tuple2[2], places=places)


if __name__ == "__main__":
    unittest.main()
