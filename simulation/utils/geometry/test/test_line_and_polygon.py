import math
import random
import unittest

import geometry_msgs.msg as g_msgs
import hypothesis.strategies as st
import numpy as np
from hypothesis import given

import simulation.utils.geometry.line as line_module
from simulation.utils.geometry.line import Line
from simulation.utils.geometry.point import Point
from simulation.utils.geometry.polygon import Polygon
from simulation.utils.geometry.pose import Pose
from simulation.utils.geometry.transform import Transform
from simulation.utils.geometry.vector import Vector

TOLERANCE = 0.007


class ModuleTest(unittest.TestCase):
    def create_points(self, count=10):
        points = []

        for _ in range(count):
            x = random.random()
            y = random.random()
            z = random.random()
            p = Point(x, y, z)

            points.append(p)

        return points

    def test_line_init(self):
        points = self.create_points()
        line = Line(points)

        g_line = [p.to_geometry_msg() for p in points]
        self.assertEqual(Line(g_line), line)

        np_array = [p.to_numpy() for p in points]
        self.assertEqual(Line(np_array), line)

        # List of two dimensional coords
        projected_points = [Point(p.x, p.y) for p in points]
        coords_2d = [[p.x, p.y] for p in points]

        self.assertEqual(Line(projected_points), Line(coords_2d))

    def test_line_extract(self):

        points = self.create_points()
        line = Line(points)

        self.assertListEqual(points, line.get_points())

        g_line = [p.to_geometry_msg() for p in points]
        self.assertListEqual(g_line, line.to_geometry_msgs())

        np_array = np.array([p.to_numpy() for p in points])
        self.assertEqual(np_array.all(), line.to_numpy().all())

    def test_line_interpolation_func(self):
        line = Line([Point(0, 0), Point(1, 1)])

        # Test if line direction function works
        self.assertAlmostEqual(
            line.interpolate_direction(arc_length=0),
            Vector(r=1, phi=math.pi / 4),
            delta=TOLERANCE,
        )
        self.assertAlmostEqual(
            line.interpolate_direction(arc_length=line.length / 2),
            Vector(r=1, phi=math.pi / 4),
            delta=TOLERANCE,
        )
        self.assertAlmostEqual(
            line.interpolate_direction(arc_length=line.length),
            Vector(r=1, phi=math.pi / 4),
            delta=TOLERANCE,
        )

        self.assertAlmostEqual(line.interpolate_curvature(arc_length=0), 0, delta=TOLERANCE)
        self.assertAlmostEqual(
            line.interpolate_curvature(arc_length=line.length / 4), 0, delta=TOLERANCE
        )

        self.assertEqual(
            line.interpolate_pose(arc_length=0), Pose(Point(0, 0), Vector(1, 1, 0))
        )

        circle = Line(
            reversed([p for p in Point(-1, 0).buffer(1, resolution=4096).exterior.coords])
        )

        # Test if line direction function works
        self.assertAlmostEqual(
            circle.interpolate_direction(arc_length=0), Vector(0, 1), delta=TOLERANCE
        )
        self.assertAlmostEqual(
            circle.interpolate_direction(arc_length=circle.length / 2),
            Vector(0, -1),
            delta=TOLERANCE,
        )
        self.assertAlmostEqual(
            circle.interpolate_direction(arc_length=circle.length),
            Vector(0, 1),
            delta=TOLERANCE,
        )
        self.assertAlmostEqual(
            circle.interpolate_direction(arc_length=circle.length * 3 / 4), Vector(1, 0)
        )
        self.assertAlmostEqual(
            circle.interpolate_direction(arc_length=circle.length / 4), Vector(-1, 0)
        )

        self.assertAlmostEqual(
            circle.interpolate_curvature(arc_length=0), 1, delta=TOLERANCE
        )
        self.assertAlmostEqual(
            circle.interpolate_curvature(arc_length=circle.length), 1, delta=TOLERANCE
        )

        self.assertAlmostEqual(
            circle.interpolate_curvature(arc_length=circle.length / 2), 1, delta=TOLERANCE
        )
        self.assertAlmostEqual(
            circle.interpolate_curvature(arc_length=circle.length / 4), 1, delta=TOLERANCE
        )

        short_line = Line(
            [
                [0, 0],
                [line_module.CURVATURE_APPROX_DISTANCE * 0.5, 0],
                [
                    line_module.CURVATURE_APPROX_DISTANCE * 0.5,
                    line_module.CURVATURE_APPROX_DISTANCE * 0.5,
                ],
            ]
        )
        with self.assertRaises(ValueError):
            short_line.interpolate_curvature(0)

        just_long_enough_line = Line(
            [[0, 0], [2 * line_module.CURVATURE_APPROX_DISTANCE, 0]]
        )
        self.assertEqual(just_long_enough_line.interpolate_curvature(0), 0)

        def assert_approx_equal_pose(pose1, pose2):
            """Substitute self.assertAlmostEqual because pose - pose is not implemented."""
            self.assertAlmostEqual(Vector(pose1.position), Vector(pose2.position))
            self.assertAlmostEqual(
                pose1.get_angle(), pose2.get_angle(), delta=math.radians(0.02)
            )

        assert_approx_equal_pose(
            circle.interpolate_pose(arc_length=0), Pose(Point(0, 0), Vector([0, 1, 0]))
        )
        assert_approx_equal_pose(
            circle.interpolate_pose(arc_length=circle.length / 4),
            Pose(Point(-1, 1), Vector([-1, 0, 0])),
        )
        assert_approx_equal_pose(
            circle.interpolate_pose(arc_length=circle.length / 2),
            Pose(Point(-2, 0), Vector([0, -1, 0])),
        )
        assert_approx_equal_pose(
            circle.interpolate_pose(arc_length=circle.length * 3 / 4),
            Pose(Point(-1, -1), Vector([1, 0, 0])),
        )
        assert_approx_equal_pose(
            circle.interpolate_pose(arc_length=circle.length),
            Pose(Point(0, 0), Vector([0, 1, 0])),
        )

    def test_line_func(self):
        points = self.create_points(10)
        line = Line()

        line += Line(points[:5])
        line += Line(points[5:])

        self.assertEqual(Line(points), line)

        line = Line([x, x] for x in range(10))
        l_r = Line([x + math.sqrt(2), x - math.sqrt(2)] for x in range(10))
        l_p = line.parallel_offset(2, "right")
        self.assertEqual(l_p, l_r, f"Lines {l_r} and {l_p} are not equal.")

        tf = Transform(Point(1, 0, 2), math.pi / 2)

        self.assertEqual(tf * line, Line([1 - x, x, 2] for x in range(10)))

    @given(
        point_count=st.integers(min_value=0, max_value=200),
        cut_length=st.floats(min_value=0, max_value=100000),
    )
    def test_line_cut(self, point_count: int, cut_length: float):
        """Test the line's class cut function.

        The test creates a line, cuts it into two and adjoins the two lines again. The
        resulting line must be the same as the initial line. Additionally, some sanity
        checks are performed.
        """
        if point_count == 0 and cut_length > 0 or point_count == 1:
            return
        initial_line = Line(self.create_points(point_count))

        if initial_line.length > 0:
            cut_length %= initial_line.length

        line1, line2 = Line.cut(initial_line, arc_length=cut_length)

        # Assert start and end points:
        if point_count > 0:
            self.assertEqual(
                (line1 if cut_length > 0 else line2).get_points()[0],
                initial_line.get_points()[0],
            )
            self.assertEqual(
                (line2 if cut_length < initial_line.length else line1).get_points()[-1],
                initial_line.get_points()[-1],
            )

        # assert line lengths
        self.assertAlmostEqual(initial_line.length, line1.length + line2.length)
        self.assertAlmostEqual(line1.length, cut_length)
        self.assertAlmostEqual(line2.length, initial_line.length - cut_length)

        # assert composition
        stitched = line1 + line2
        for p in initial_line.get_points():
            self.assertLess(stitched.distance(p), 0.0001)
        for p in stitched.get_points():
            self.assertLess(initial_line.distance(p), 0.0001)

    def test_polygon_init(self):
        points = self.create_points()
        poly = Polygon(points)

        # Check that polygon contains points
        # Shapely adds first point at last index again
        self.assertListEqual(poly.get_points()[:-1], points)

        # Geometry msgs
        g_polygon = g_msgs.Polygon()
        g_polygon.points = [p.to_geometry_msg() for p in points]
        poly_g = Polygon(g_polygon)
        self.assertEqual(poly, poly_g)

        # Numpy
        array = np.array([p.to_numpy() for p in points])
        poly_np = Polygon(array)
        self.assertEqual(poly, poly_np)

        # Check if polygon can be created correctly from array of two dimensional points
        points_2d = [[2, 1], [3, 2], [34, 5], [3242, 1]]
        points_3d = [[*p, 0] for p in points_2d]
        self.assertEqual(Polygon(points_2d), Polygon(points_3d))

        # Two lines
        points_left = points
        points_right = [(p + Vector(1, 0)) for p in points]

        all_points = points_right.copy()
        all_points.extend(reversed(points_left))  # Left side in reverse

        poly1 = Polygon(Line(points_left), Line(points_right))
        poly2 = Polygon(all_points)
        self.assertEqual(poly1, poly2)

    def test_polygon_extract(self):
        """Test if the polygon methods extraction functions work."""
        points = self.create_points()

        poly = Polygon(points)

        g_polygon = g_msgs.Polygon()
        g_polygon.points = [p.to_geometry_msg() for p in points]
        # append first point at the end to match behaviour of Polygon
        g_polygon.points.append(points[0].to_geometry_msg())

        array = np.array([p.to_numpy() for p in points])

        self.assertEqual(poly.to_geometry_msg(), g_polygon)
        self.assertEqual(poly.to_numpy().all(), array.all())

    def test_polygon_func(self):
        """Test polygon functions."""

        poly = Polygon([Point(1, 0), Point(2, 1), Point(2, 2), Point(-1, 0, 2)])

        # Rotate by 90 degrees -> [0,1],[-1,2],[-2,2],[0,-1,2]
        # Translate by 1,0,2 -> [1,1,2],[0,2,2],[-1,2,2],[1,-1,4]
        tf = Transform(Point(1, 0, 2), math.pi / 2)

        self.assertEqual(
            tf * poly,
            Polygon([Point(1, 1, 2), Point(0, 2, 2), Point(-1, 2, 2), Point(1, -1, 4)]),
        )

        # Test polygon eq function
        poly_rev = Polygon(
            [
                Point(1, 1, 2),
                Point(1, -1, 4),
                Point(-1, 2, 2),
                Point(0, 2, 2),
                Point(1, 1, 2),
            ]
        )
        poly_uneq = Polygon([Point(2, 2), Point(2, 1), Point(1, 0)])

        self.assertEqual(poly, poly)
        self.assertEqual(tf * poly, poly_rev)
        self.assertNotEqual(tf * poly, poly_uneq)


if __name__ == "__main__":
    unittest.main()
