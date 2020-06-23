import unittest
import math
import random

from simulation.utils.geometry.point import Point
from simulation.utils.geometry.vector import Vector
from simulation.utils.geometry.line import Line
from simulation.utils.geometry.polygon import Polygon
from simulation.utils.geometry.pose import Pose
from simulation.utils.geometry.transform import Transform

from pyquaternion import Quaternion

import geometry_msgs.msg as g_msgs


class ModuleTest(unittest.TestCase):
    def assert_equal_angles(self, angle1, angle2):
        a1 = (angle1 - 2 * int(angle1) * math.pi) % (2 * math.pi)
        a2 = (angle2 - 2 * int(angle2) * math.pi) % (2 * math.pi)
        self.assertAlmostEqual(a1, a2)

    def create_points(self, count=10):

        points = []

        for _ in range(count):
            x = random.random()
            y = random.random()
            z = random.random()
            p = Point(x, y, z)
            points.append(p)

        return points

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
        o = Quaternion(math.sqrt(1 / 2), 0, 0, math.sqrt(1 / 2))
        tf = Transform(v, o)

        # Create from angle
        tf2 = Transform(v, math.pi / 2)

        self.assertEqual(tf, tf2)

        # Create from geometry_msgs/transform
        g_tf = g_msgs.Transform()
        g_tf.translation = v.to_geometry_msg()
        g_tf.rotation = g_msgs.Quaternion(0, 0, math.sqrt(1 / 2), math.sqrt(1 / 2))

        self.assertEqual(Transform(g_tf), tf)

    def test_tf_funcs(self):
        v = Vector(1, 3, 2)
        # Create from angle
        tf = Transform(v, math.pi / 2)

        self.assertAlmostEqual(tf.get_angle(), math.pi / 2)

        # Extensively test the angle function:
        for _ in range(0, 100):
            angle = random.randrange(-100, 100)
            self.assert_equal_angles(Transform(v, angle).get_angle(), angle)

        g_tf = g_msgs.Transform()
        g_tf.translation = v.to_geometry_msg()
        g_tf.rotation = g_msgs.Quaternion(0, 0, math.sqrt(1 / 2), math.sqrt(1 / 2))

        self.geom_tf_almost_eq(g_tf, tf.to_geometry_msg())

        # Should return 90 degrees
        tf2 = Transform(Vector(1, 0, 0), Quaternion(1, 0, 0, 1).normalised)
        self.assertEqual(tf2.get_angle(), math.pi / 2)

        # Multiply transforms
        self.assertEqual(tf * tf2, Transform(Vector(1, 4, 2), math.pi))

    def test_tf_inverse(self):
        # Extensively test the inverse function:
        for _ in range(0, 100):
            tf = Transform(
                [random.random(), random.random(), random.random()], random.random()
            )
            self.assertEqual(tf * tf.inverse, Transform([0, 0, 0], 0))
            self.assertEqual(tf.inverse * tf, Transform([0, 0, 0], 0))

    def test_transformations(self):

        tf = Transform(Vector(2, 2), math.pi / 2)

        self.assertEqual(Vector(1, 3), tf * Vector(1, 1))
        self.assertEqual(Point(1, 3), tf * Point(1, 1))

        # Check if line and polygon are transformed correctly
        points = self.create_points()
        transformed_points = [tf * p for p in points]

        self.assertEqual(tf * Line(points), Line(transformed_points))
        self.assertEqual(tf * Polygon(points), Polygon(transformed_points))

        # Check to transform a pose
        pose = Pose(Point(2, 2), math.pi)
        self.assertEqual(tf * pose, Pose([0, 4], math.pi * 3 / 2))


if __name__ == "__main__":
    random.seed("KITCAR")
    unittest.main()
