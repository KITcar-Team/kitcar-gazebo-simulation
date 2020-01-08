import unittest
import random
import math

from base.point import Point
from base.point import InvalidPointOperationError
from base.vector import Vector
from base.line import Line
from base.polygon import Polygon
from base.pose import Pose
from base.transform import Transform

from pyquaternion import Quaternion

import geometry_msgs.msg as g_msgs
import numpy as np
import road_generation.schema as schema


class ModuleTest(unittest.TestCase):

    def test_vector_init(self):
        """ Test if the vector class can be initialize. """

        # Basic
        p1 = Vector(1, 3)
        self.assertListEqual([p1.x, p1.y, p1.z], [1, 3, 0])

        examples = [([-1, -2], [-1, -2, 0]),
                    (np.array([1, 2, 4]), [1, 2, 4]),
                    (np.array([1, 2]), [1, 2, 0]),
                    (g_msgs.Vector3(1, 3, 4), [1, 3, 4])]

        for example in examples:
            # Point initialization
            v = Vector(example[0])
            self.assertListEqual([v.x, v.y, v.z], example[1])

        #r, phi
        self.assertEqual(Vector(math.sqrt(2), math.sqrt(2)),
                         Vector(r=2, phi=math.pi/4))

    def test_vector_extract(self):
        p1 = Vector(1, 3, 2)

        self.assertEqual(p1.to_numpy().all(), np.array([1, 3, 2]).all())
        self.assertEqual(p1.to_geometry_msg(), g_msgs.Vector3(1, 3, 2))

    def test_vector_func(self):
        p1 = Vector(1, 2, 3)
        p2 = Vector(2, 1, 3)
        p3 = Vector(3, 3, 6)

        self.assertEqual(p1+p2, p3)
        self.assertEqual(p3-p2, p1)
        self.assertEqual(3*p1, Vector(3, 6, 9))

        self.assertEqual(p1.rotated(math.pi/2), Vector(-2, 1, 3))

    def test_point_init(self):
        """ Test if the point class can be initialize. """

        # Basic
        p1 = Point(1, 3)
        self.assertListEqual([p1.x, p1.y, p1.z], [1, 3, 0])

        examples = [([-1, -2], [-1, -2, 0]),
                    (np.array([1, 2, 4]), [1, 2, 4]),
                    (np.array([1, 2]), [1, 2, 0]),
                    (g_msgs.Point(1, 3, 4), [1, 3, 4])]

        for example in examples:
            # Point initialization
            point = Point(example[0])
            self.assertListEqual([point.x, point.y, point.z], example[1])

        self.assertEqual(Point(math.sqrt(2), math.sqrt(2)),
                         Point(r=2, phi=math.pi/4))

    def test_point_extract(self):
        p1 = Point(1, 3, 2)

        self.assertEqual(p1.to_geometry_msg(), g_msgs.Point(1, 3, 2))
        self.assertEqual(p1.to_schema(), schema.point(x=1, y=3))

    def test_point_func(self):
        p1 = Point(1, 2, 3)
        vec = Vector(2, 1, 3)
        p3 = Point(3, 3, 6)

        self.assertEqual(p1+vec, p3)
        self.assertEqual(p3-vec, p1)

        # The following should raise exception
        with self.assertRaises(InvalidPointOperationError):
            p1.rotated(0)

        with self.assertRaises(InvalidPointOperationError):
            p1 + p3

        with self.assertRaises(InvalidPointOperationError):
            p1 - p3

        with self.assertRaises(InvalidPointOperationError):
            3*p3

######## TRANSFORM ########

    def geom_tf_almost_eq(self, p1, p2):

        self.assertAlmostEqual(p1.translation, p2.translation)

        self.assertAlmostEqual(p1.rotation.w, p2.rotation.w)
        self.assertAlmostEqual(p1.rotation.x, p2.rotation.x)
        self.assertAlmostEqual(p1.rotation.y, p2.rotation.y)
        self.assertAlmostEqual(p1.rotation.z, p2.rotation.z)

    def test_tf_init(self):
        """ Test if tf class can be initialized as expected. """
        # Create from quaternion
        v = Vector(1, 3, 2)
        o = Quaternion(math.sqrt(1/2), 0, 0, math.sqrt(1/2))
        tf = Transform(v, o)

        # Create from angle
        tf2 = Transform(v, math.pi/2)

        self.assertEqual(tf, tf2)

        # Create from geometry_msgs/transform
        g_tf = g_msgs.Transform()
        g_tf.translation = v.to_geometry_msg()
        g_tf.rotation = g_msgs.Quaternion(0, 0, math.sqrt(1/2), math.sqrt(1/2))

        self.assertEqual(Transform(g_tf), tf)

    def test_tf_funcs(self):
        v = Vector(1, 3, 2)
        # Create from angle
        tf = Transform(v, math.pi/2)

        self.assertAlmostEqual(tf.get_angle(), math.pi/2)

        g_tf = g_msgs.Transform()
        g_tf.translation = v.to_geometry_msg()
        g_tf.rotation = g_msgs.Quaternion(
            0, 0, math.sqrt(1/2), math.sqrt(1/2))

        self.geom_tf_almost_eq(g_tf, tf.to_geometry_msg())

        # Should return 90 degrees
        tf2 = Transform(Vector(1, 0, 0), Quaternion(1, 0, 0, 1).normalised)
        self.assertEqual(tf2.get_angle(), math.pi/2)

        # Multiply transforms
        self.assertEqual(tf*tf2, Transform(Vector(1, 4, 2), math.pi))

######## POSE ########

    def geom_pose_almost_eq(self, p1, p2):

        self.assertAlmostEqual(p1.position, p2.position)

        self.assertAlmostEqual(p1.orientation.w, p2.orientation.w)
        self.assertAlmostEqual(p1.orientation.x, p2.orientation.x)
        self.assertAlmostEqual(p1.orientation.y, p2.orientation.y)
        self.assertAlmostEqual(p1.orientation.z, p2.orientation.z)

    def test_pose_init(self):
        """ Test if pose class can be initialized as expected. """
        # Create from quaternion
        p = Point(1, 3, 2)
        o = Quaternion(math.sqrt(1/2), 0, 0, math.sqrt(1/2))
        pose = Pose(p, o)

        # Create from angle
        pose2 = Pose(p, math.pi/2)

        self.assertEqual(pose, pose2)

        g_pose = g_msgs.Pose()
        g_pose.position = p.to_geometry_msg()
        g_pose.orientation = g_msgs.Quaternion(
            0, 0, math.sqrt(1/2), math.sqrt(1/2))

        self.assertEqual(Pose(g_pose), pose)

        # Create from geometry_msgs/transform
        g_tf = g_msgs.Transform()
        g_tf.translation = p.to_geometry_msg()
        g_tf.rotation = g_msgs.Quaternion(0, 0, math.sqrt(1/2), math.sqrt(1/2))

        self.assertEqual(Pose(g_tf), pose)

    def test_pose_funcs(self):
        p = Point(1, 3, 2)
        # Create from angle
        pose = Pose(p, math.pi/2)

        self.assertAlmostEqual(pose.get_angle(), math.pi/2)

        g_pose = g_msgs.Pose()
        g_pose.position = p.to_geometry_msg()
        g_pose.orientation = g_msgs.Quaternion(
            0, 0, math.sqrt(1/2), math.sqrt(1/2))

        self.geom_pose_almost_eq(g_pose, pose.to_geometry_msg())

        # Should return 90 degrees
        pose2 = Pose(Point(1, 0, 0), Quaternion(-1, 0, 0, -1).normalised)
        self.assertEqual(pose2.get_angle(), math.pi/2)

        # Apply transformations
        tf2 = Transform(Vector(1, 0, 0), Quaternion(1, 0, 0, 1).normalised)

        # Multiply transforms
        self.assertEqual(pose*tf2, Pose(Vector(1, 4, 2), math.pi))

        self.assertEqual(pose / tf2, Pose(Vector(1, 2, 2), 0))


##### Helper function #####


    def create_points(self, count=10):
        points = []

        for _ in range(count):
            x = random.random()
            y = random.random()
            z = random.random()
            p = Point(x, y, z)

            points.append(p)

        return points

######### LINE #########

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

        boundary = schema.boundary()
        boundary.point = [p.to_schema() for p in points]
        self.assertEqual(boundary, line.to_schema_boundary())

    def test_line_func(self):
        points = self.create_points(10)
        line = Line()

        line += Line(points[:5])
        line += Line(points[5:])

        self.assertEqual(Line(points), line)

        line = Line([x, x] for x in range(10))
        l_r = Line([x + math.sqrt(2), x - math.sqrt(2)] for x in range(10))
        l_p = line.parallel_offset(2, 'right')
        self.assertEqual(l_p, l_r)

        tf = Transform(Point(1, 0, 2), math.pi/2)

        self.assertEqual(tf*line, Line([1-x,x,2] for x in range(10)))

######### POLYGON #########

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
        """ Test if the polygon methods extraction functions work."""
        points = self.create_points()

        poly = Polygon(points)

        g_polygon = g_msgs.Polygon()
        g_polygon.points = [p.to_geometry_msg() for p in points]

        array = np.array([p.to_numpy() for p in points])

        # self.assertEqual(poly.to_geometry_msg(),g_polygon)
        self.assertEqual(poly.to_numpy().all(), array.all())

        # Test if lanelet is created correctly
        left_points = self.create_points(10)
        right_points = self.create_points(10)

        lanelet = schema.lanelet()
        lanelet.leftBoundary = Line(left_points).to_schema_boundary()
        lanelet.rightBoundary = Line(right_points).to_schema_boundary()

        # Points for polygon
        poly_points = right_points
        poly_points.extend(reversed(left_points))

        p = Polygon(poly_points)
        self.assertEqual(p.to_schema_lanelet(), lanelet)

    def test_polygon_func(self):
        """Test polygon functions."""

        poly = Polygon([Point(1, 0), Point(2, 1),
                        Point(2, 2), Point(-1, 0, 2)])

        # Rotate by 90 degrees -> [0,1],[-1,2],[-2,2],[0,-1,2]
        # Translate by 1,0,2 -> [1,1,2],[0,2,2],[-1,2,2],[1,-1,4]
        tf = Transform(Point(1, 0, 2), math.pi/2)

        self.assertEqual(tf*poly, Polygon(
            [Point(1, 1, 2), Point(0, 2, 2), Point(-1, 2, 2), Point(1, -1, 4)]))

        # Test polygon eq function
        poly_rev = Polygon([Point(1, 1, 2), Point(1, -1, 4),
                            Point(-1, 2, 2), Point(0, 2, 2), Point(1, 1, 2)])
        poly_uneq = Polygon([Point(2, 2), Point(2, 1), Point(1, 0)])

        self.assertEqual(poly, poly)
        self.assertTrue(tf*poly == poly_rev)
        self.assertFalse(tf*poly == poly_uneq)

    def test_transformations(self):
            
        tf = Transform(Vector(2,2),math.pi / 2)
        
        self.assertEqual(Vector(1,3),tf*Vector(1,1))
        self.assertEqual(Point(1,3),tf*Point(1,1))


if __name__ == '__main__':
    unittest.main()
