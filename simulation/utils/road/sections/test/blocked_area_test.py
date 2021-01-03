import math
import unittest

import simulation.utils.road.sections.type as road_section_type
from simulation.utils.geometry import Point, Polygon, Transform
from simulation.utils.road.config import Config
from simulation.utils.road.sections.blocked_area import BlockedArea


class ModuleTest(unittest.TestCase):
    def test_blocked_area(self):
        TF = Transform([1, 1], 0)
        LENGTH = 2.5
        WIDTH = 0.3
        OPENING_ANGLE = math.radians(60)

        ba = BlockedArea(length=LENGTH, width=WIDTH)
        ba.set_transform(TF)
        self.assertEqual(ba.__class__.TYPE, road_section_type.BLOCKED_AREA)
        inset = WIDTH / math.tan(OPENING_ANGLE)
        self.assertEqual(
            ba.frame,
            TF
            * Polygon(
                [
                    Point(0, -Config.road_width),
                    Point(inset, -Config.road_width + WIDTH),
                    Point(LENGTH - inset, -Config.road_width + WIDTH),
                    Point(LENGTH, -Config.road_width),
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
