import unittest
import math

from simulation.utils.geometry.vector import Vector

import geometry_msgs.msg as g_msgs
import numpy as np

import random


class ModuleTest(unittest.TestCase):
    def test_vector_init(self):
        """ Test if the vector class can be initialize. """

        # Basic
        p1 = Vector(1, 3)
        self.assertListEqual([p1.x, p1.y, p1.z], [1, 3, 0])

        examples = [
            ([-1, -2], [-1, -2, 0]),
            (np.array([1, 2, 4]), [1, 2, 4]),
            (np.array([1, 2]), [1, 2, 0]),
            (g_msgs.Vector3(1, 3, 4), [1, 3, 4]),
        ]

        for example in examples:
            # Point initialization
            v = Vector(example[0])
            self.assertListEqual([v.x, v.y, v.z], example[1])

        # r, phi
        self.assertEqual(Vector(math.sqrt(2), math.sqrt(2)), Vector(r=2, phi=math.pi / 4))

    def test_vector_extract(self):
        p1 = Vector(1, 3, 2)

        self.assertEqual(p1.to_numpy().all(), np.array([1, 3, 2]).all())
        self.assertEqual(p1.to_geometry_msg(), g_msgs.Vector3(1, 3, 2))

    def test_vector_func(self):
        p1 = Vector(1, 2, 3)
        p2 = Vector(2, 1, 3)
        p3 = Vector(3, 3, 6)

        self.assertEqual(p1 + p2, p3)
        self.assertEqual(p3 - p2, p1)
        self.assertEqual(3 * p1, Vector(3, 6, 9))

        self.assertEqual(p1.rotated(math.pi / 2), Vector(-2, 1, 3))

        # Test norm function and scalar product
        self.assertEqual(p1 * p3, p1.x * p3.x + p1.y * p3.y + p1.z * p3.z)
        self.assertEqual(p1 * p1, abs(p1) * abs(p1))
        self.assertEqual(p1 * (p2 - p3), p1 * p2 - p1 * p3)
        self.assertEqual(p1 * p2, p2 * p1)

        # Test cross product
        self.assertAlmostEqual(p1.cross(p2), Vector(3, 3, -3))

    def test_hash_func(self):
        """Check random points that their hash is equal iff they are equal.

        Check in (-100,100).
        """

        v1 = Vector(0, 0, 0)
        v2 = Vector(0, 0, 0)
        for _ in range(1000):
            v1.__hash__()
            x = (random.random() - 0.5) * 200
            y = (random.random() - 0.5) * 200
            z = (random.random() - 0.5) * 200
            v2 = Vector(x, y, z)
            self.assertEqual(v1.__hash__() == v2.__hash__(), v1 == v2)
            v1 = v2


if __name__ == "__main__":
    random.seed("KITCAR")
    unittest.main()
