import unittest
import math

from simulation.utils.geometry import Line, Transform
import simulation.utils.road.sections.type as road_section_type
from simulation.utils.road.sections import StraightRoad


class ModuleTest(unittest.TestCase):
    def test_straight_road(self):
        TF = Transform([1, 1], math.pi / 2)
        LENGTH = 4
        MIDDLE_LINE = Line([[1, 1], [1, 5]])

        sr = StraightRoad(length=LENGTH, transform=TF)
        self.assertEqual(sr.__class__.TYPE, road_section_type.STRAIGHT_ROAD)
        self.assertEqual(sr.middle_line, MIDDLE_LINE)


if __name__ == "__main__":
    unittest.main()
