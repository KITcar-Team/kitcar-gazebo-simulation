import unittest
import math

from simulation.utils.geometry.point import Point
from simulation.utils.geometry.point import InvalidPointOperationError
from simulation.utils.geometry.vector import Vector

import geometry_msgs.msg as g_msgs
import numpy as np


class ModuleTest(unittest.TestCase):
    def test_point_init(self):
        """ Test if the point class can be initialize. """

        # Basic
        p1 = Point(1, 3)
        self.assertListEqual([p1.x, p1.y, p1.z], [1, 3, 0])

        examples = [
            ([-1, -2], [-1, -2, 0]),
            (np.array([1, 2, 4]), [1, 2, 4]),
            (np.array([1, 2]), [1, 2, 0]),
            (g_msgs.Point32(1, 3, 4), [1, 3, 4]),
        ]

        for example in examples:
            # Point initialization
            point = Point(example[0])
            self.assertListEqual([point.x, point.y, point.z], example[1])

        self.assertEqual(Point(math.sqrt(2), math.sqrt(2)), Point(r=2, phi=math.pi / 4))

    def test_point_extract(self):
        p1 = Point(1, 3, 2)

        self.assertEqual(p1.to_geometry_msg(), g_msgs.Point32(1, 3, 2))

    def test_point_func(self):
        p1 = Point(1, 2, 3)
        vec = Vector(2, 1, 3)
        p3 = Point(3, 3, 6)

        self.assertEqual(p1 + vec, p3)
        self.assertEqual(p3 - vec, p1)

        # The following should raise exception
        with self.assertRaises(InvalidPointOperationError):
            p1.rotated(0)

        with self.assertRaises(InvalidPointOperationError):
            p1 + p3

        with self.assertRaises(InvalidPointOperationError):
            p1 - p3

        with self.assertRaises(InvalidPointOperationError):
            3 * p3


if __name__ == "__main__":
    unittest.main()
