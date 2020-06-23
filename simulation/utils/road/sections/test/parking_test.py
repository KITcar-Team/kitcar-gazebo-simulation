import unittest
import math

from simulation.utils.geometry import Point, Transform, Line, Polygon
import simulation.utils.road.sections.type as road_section_type
from simulation.utils.road.sections import (
    ParkingArea,
    ParkingLot,
    ParkingSpot,
)

from simulation.utils.road.config import Config


class ModuleTest(unittest.TestCase):
    # basic parking spot test
    # obstacles on spots are not tested
    # most test points are calculated by hand -> magic numbers
    def test_parking_area(self):
        TF = Transform([1, 1], math.pi / 2)
        left_spot1 = ParkingSpot(width=0.4, kind=ParkingSpot.FREE)
        left_spot2 = ParkingSpot(width=0.4, kind=ParkingSpot.OCCUPIED)
        right_spot = ParkingSpot(width=0.6, kind=ParkingSpot.BLOCKED)

        # opening_angle 55 degrees
        OPENING_ANGLE = math.radians(55)
        LEFT_START = 2.25
        LEFT_DEPTH = 0.5
        RIGHT_START = 1.75
        RIGHT_DEPTH = 0.3
        LENGTH = 4

        left_lot = ParkingLot(
            start=LEFT_START,
            opening_angle=OPENING_ANGLE,
            depth=LEFT_DEPTH,
            spots=[left_spot1, left_spot2],
        )
        right_lot = ParkingLot(
            start=RIGHT_START,
            opening_angle=OPENING_ANGLE,
            depth=RIGHT_DEPTH,
            spots=[right_spot],
        )

        pa = ParkingArea(
            length=LENGTH, left_lots=[left_lot], right_lots=[right_lot], transform=TF,
        )
        self.assertEqual(pa.__class__.TYPE, road_section_type.PARKING_AREA)

        # test points for left_border calculated by hand
        # left_lot start at 2.25, depth 0.5
        # start_inner
        si = Point(LEFT_START, Config.road_width)
        # start_outer
        so = Point(
            si.x + LEFT_DEPTH / math.tan(OPENING_ANGLE), Config.road_width + LEFT_DEPTH,
        )
        # end outer, both lots have 0.4 width
        eo = Point(so.x + 0.8, so.y)
        # end inner
        ei = Point(eo.x + LEFT_DEPTH / math.tan(OPENING_ANGLE), Config.road_width)
        self.assertLineAlmostEqual(pa.left_lots[0].border, TF * Line([si, so, eo, ei]))

        # check first parking spot on left side width: 0.4
        self.assertEqual(
            pa.left_lots[0].spots[0].frame,
            TF
            * Polygon(
                [Point(so.x, si.y), so, Point(so.x + 0.4, so.y), Point(so.x + 0.4, si.y)]
            ),
        )
        self.assertEqual(pa.left_lots[0].spots[0].kind, ParkingSpot.FREE)
        # check second parking spot on left side width: 0.4
        self.assertEqual(
            pa.left_lots[0].spots[1].frame,
            TF
            * Polygon(
                [Point(so.x + 0.4, si.y), Point(so.x + 0.4, so.y), eo, Point(eo.x, ei.y)]
            ),
        )
        self.assertEqual(pa.left_lots[0].spots[1].kind, ParkingSpot.OCCUPIED)

        # test points for right_border calculated by hand
        # left_lot start at 1.75, depth 0.3
        # start_inner
        si = Point(RIGHT_START, -Config.road_width)
        # start_outer
        so = Point(
            si.x + RIGHT_DEPTH / math.tan(OPENING_ANGLE), -Config.road_width - RIGHT_DEPTH,
        )
        # end outer, lot has 0.6 width
        eo = Point(so.x + 0.6, -Config.road_width - RIGHT_DEPTH)
        # end inner
        ei = Point(eo.x + RIGHT_DEPTH / math.tan(OPENING_ANGLE), -Config.road_width)
        self.assertLineAlmostEqual(pa.right_lots[0].border, TF * Line([si, so, eo, ei]))
        # check first parking spot on right side width: 0.6
        # polygon points on right side are reversed
        self.assertEqual(
            pa.right_lots[0].spots[0].frame,
            TF * Polygon([Point(eo.x, ei.y), eo, so, Point(so.x, si.y)]),
        )
        self.assertEqual(
            pa.right_lots[0].spots[0].kind, ParkingSpot.BLOCKED,
        )

    ###
    # Helper functions
    def assertPointAlmostEqual(self, p1, p2):
        self.assertAlmostEqual(p1.x, p2.x)
        self.assertAlmostEqual(p1.y, p2.y)
        self.assertAlmostEqual(p1.z, p2.z)

    def assertLineAlmostEqual(self, line1, line2):
        for p1, p2 in zip(line1.get_points(), line2.get_points()):
            self.assertPointAlmostEqual(p1, p2)


if __name__ == "__main__":
    unittest.main()
