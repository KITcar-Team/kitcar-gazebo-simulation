import unittest

from geometry import Point, Polygon, Transform

from road.sections.zebra_crossing import ZebraCrossing
import road.sections.type as road_section_type

from road.config import Config


class ModuleTest(unittest.TestCase):
    def test_zebra_crossing(self):
        TF = Transform([1, 1], 0)
        LENGTH = 2

        zc = ZebraCrossing(length=LENGTH, transform=TF)
        self.assertEqual(zc.__class__.TYPE, road_section_type.ZEBRA_CROSSING)
        self.assertEqual(
            zc.frame,
            TF
            * Polygon(
                [
                    Point(0, -Config.road_width),
                    Point(0, Config.road_width),
                    Point(LENGTH, Config.road_width),
                    Point(LENGTH, -Config.road_width),
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()