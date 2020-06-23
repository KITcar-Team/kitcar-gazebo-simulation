import unittest
import math

from simulation.utils.geometry import Point, Polygon, Transform, Line, Pose

from simulation.utils.road.sections.road_section import RoadSection
from simulation.utils.road.sections import StaticObstacle
from simulation.utils.road.config import Config


class DummyRoadSection(RoadSection):
    TYPE = "DUMMY"

    @property
    def middle_line(self):
        return DummyRoadSection.MIDDLE_LINE


class ModuleTest(unittest.TestCase):
    def assert_lines_approx_equal(self, line1, line2):
        p = Polygon(line1.get_points() + list(reversed(line2.get_points())))
        self.assertAlmostEqual(p.area, 0)

    def test_road_section(self):
        MIDDLE_LINE = Line([[0, 0], [1, 0]])
        DummyRoadSection.MIDDLE_LINE = MIDDLE_LINE
        LEFT_LINE = Line([[0, Config.road_width], [1, Config.road_width]])
        RIGHT_LINE = Line(
            [[0, -Config.road_width], [1 + Config.road_width, -Config.road_width]]
        )
        ENDING = (Pose([1, 0], 0), 0)
        BEGINNING = (Pose([0, 0], math.pi), 0)

        rs = DummyRoadSection()
        self.assertIsNotNone(rs.transform)
        self.assert_lines_approx_equal(rs.middle_line, MIDDLE_LINE)
        self.assert_lines_approx_equal(rs.right_line, RIGHT_LINE)
        self.assert_lines_approx_equal(rs.left_line, LEFT_LINE)
        self.assertTupleEqual(rs.get_beginning(), BEGINNING)
        self.assertTupleEqual(rs.get_ending(), ENDING)
        self.assertTrue(rs.get_bounding_box().contains(LEFT_LINE))
        self.assertTrue(rs.get_bounding_box().contains(MIDDLE_LINE))
        self.assertTrue(rs.get_bounding_box().contains(RIGHT_LINE))

    def test_obstacles(self):
        MIDDLE_LINE = Line([[0, 0], [0, 2]])
        DummyRoadSection.MIDDLE_LINE = MIDDLE_LINE

        # Assume the obstacle class itself works!
        # Simply test if the transforms etc. are set correctly.

        obstacle = StaticObstacle(center=Point(1.5, -0.2), width=0.2, depth=1)
        rs = DummyRoadSection(obstacles=[obstacle])
        returned_obs = rs.obstacles[0]
        self.assertEqual(returned_obs.transform, Transform([0, 1.5], math.pi / 2))
        self.assertEqual(returned_obs.center, Point(0.2, 1.5))
        self.assertEqual(
            returned_obs.frame, Polygon([[0.1, 1], [0.1, 2], [0.3, 2], [0.3, 1]])
        )

        #
        # Test obstacle with angle 90 degrees and
        # Transform(Vector(0, 0), 90°)
        # obs_args.update({"angle": 90})
        # test_transform = Transform(Vector(1, 1), 0.5 * math.pi)

        # First translate mid_point to (0, 0) and rotate by angle
        # test_angle = 0.5 * math.pi
        # test_left_lower = (test_left_lower - test_mid).rotated(test_angle) + test_mid
        # test_left_upper = (test_left_upper - test_mid).rotated(test_angle) + test_mid
        # test_right_lower = (test_right_lower - test_mid).rotated(test_angle) + test_mid
        # test_right_upper = (test_right_upper - test_mid).rotated(test_angle) + test_mid

        # Second transform entire polygon
        # test_poly = test_transform * Polygon(
        #    [test_left_lower, test_left_upper, test_right_upper, test_right_lower,]
        # )
        # transform test mid
        # test_mid = test_transform * test_mid

        # construct obstacle and set transform
        # obstacle = StaticObstacle(obs_args)
        # transform x-value of left_lower to use as transform for obstacle
        # p1 = test_transform * Vector(obstacle._left_lower_x, 0)
        # obstacle.transform = Transform(p1, 0.5 * math.pi)
        # gt = obstacle.generate_groundtruth()

        # compare obstacles
        # self.assertPolygonAlmostEqual(gt, test_poly)

        #
        # Helper functions
        # def assertPointAlmostEqual(self, p1, p2):
        #    self.assertAlmostEqual(p1.x, p2.x)
        #    self.assertAlmostEqual(p1.y, p2.y)
        #    self.assertAlmostEqual(p1.z, p2.z)

        # def assertPolygonAlmostEqual(self, poly1, poly2):
        #    for p1, p2 in zip(poly1.get_points(), poly2.get_points()):
        #        self.assertPointAlmostEqual(p1, p2)


if __name__ == "__main__":
    unittest.main()
