import unittest
import math

from simulation.utils.geometry.point import Point
from simulation.utils.geometry.vector import Vector
from simulation.utils.geometry.pose import Pose
from simulation.utils.geometry.transform import Transform

from pyquaternion import Quaternion

import geometry_msgs.msg as g_msgs


class ModuleTest(unittest.TestCase):
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
        o = Quaternion(math.sqrt(1 / 2), 0, 0, math.sqrt(1 / 2))
        pose = Pose(p, o)

        # Create from angle
        pose2 = Pose(p, math.pi / 2)

        self.assertEqual(pose, pose2)

        g_pose = g_msgs.Pose()
        g_pose.position = p.to_geometry_msg()
        g_pose.orientation = g_msgs.Quaternion(0, 0, math.sqrt(1 / 2), math.sqrt(1 / 2))

        self.assertEqual(Pose(g_pose), pose)

        # Create from geometry_msgs/transform
        g_tf = g_msgs.Transform()
        g_tf.translation = p.to_geometry_msg()
        g_tf.rotation = g_msgs.Quaternion(0, 0, math.sqrt(1 / 2), math.sqrt(1 / 2))

        self.assertEqual(Pose(g_tf), pose)

    def test_pose_funcs(self):
        p = Point(1, 3, 2)
        # Create from angle
        pose = Pose(p, math.pi / 2)

        self.assertAlmostEqual(pose.get_angle(), math.pi / 2)

        g_pose = g_msgs.Pose()
        g_pose.position = p.to_geometry_msg()
        g_pose.orientation = g_msgs.Quaternion(0, 0, math.sqrt(1 / 2), math.sqrt(1 / 2))

        self.geom_pose_almost_eq(g_pose, pose.to_geometry_msg())

        # Should return 90 degrees
        pose2 = Pose(Point(1, 0, 0), Quaternion(-1, 0, 0, -1).normalised)
        self.assertEqual(pose2.get_angle(), math.pi / 2)

        # Apply transformations
        tf2 = Transform(Vector(1, 0, 0), Quaternion(1, 0, 0, 1).normalised)

        # Multiply transforms
        self.assertEqual(tf2 * pose, Pose(Vector(1, 4, 2), math.pi))


if __name__ == "__main__":
    unittest.main()
