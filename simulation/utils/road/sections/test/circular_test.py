import unittest
import math

from simulation.utils.geometry import Point, Transform, Vector
import simulation.utils.road.sections.type as road_section_type
from simulation.utils.road.sections import (
    LeftCircularArc,
    RightCircularArc,
)


class ModuleTest(unittest.TestCase):
    def assert_line_is_arc(self, line, center, radius, angle):
        self.assertAlmostEqual(line.length, radius * angle, delta=0.001)

        for p in (Transform(-1 * Vector(center), 0) * line).get_points():
            self.assertAlmostEqual(abs(p), radius)

    def test_left_circular_arc(self):
        TF = Transform([1, 1], math.pi / 2)
        RADIUS = 1
        ANGLE = math.pi / 2

        lca = LeftCircularArc(radius=RADIUS, angle=ANGLE, transform=TF)
        self.assertEqual(lca.__class__.TYPE, road_section_type.LEFT_CIRCULAR_ARC)
        self.assertAlmostEqual(lca.middle_line.get_points()[0], Point(TF))
        self.assertAlmostEqual(lca.middle_line.get_points()[-1], Point(0, 2))
        self.assert_line_is_arc(lca.middle_line, Point(0, 1), RADIUS, ANGLE)

    def test_right_circular_arc(self):
        TF = Transform([1, 1], math.pi / 2)
        RADIUS = 1
        ANGLE = math.pi / 2

        rca = RightCircularArc(radius=RADIUS, angle=ANGLE, transform=TF)
        self.assertEqual(rca.__class__.TYPE, road_section_type.RIGHT_CIRCULAR_ARC)
        self.assertAlmostEqual(rca.middle_line.get_points()[0], Point(TF))
        self.assertAlmostEqual(rca.middle_line.get_points()[-1], Point(2, 2))
        self.assert_line_is_arc(rca.middle_line, Point(2, 1), RADIUS, ANGLE)


if __name__ == "__main__":
    unittest.main()
