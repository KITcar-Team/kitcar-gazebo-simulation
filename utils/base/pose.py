# -*- coding: utf-8 -*-
"""Pose."""

# Compatible formats
import geometry_msgs.msg as geometry_msgs
from base.vector import Vector
from base.point import Point
from base.transform import Transform
from pyquaternion import Quaternion

import numbers
import math
import numpy as np

from contextlib import suppress

__author__ = "Konstantin Ditschuneit"
__copyright__ = "KITcar"


class Pose(Point):
    """Pose class consisting of a position and an orientation.

    A Pose object can be used to describe the state of an object with a position and an orientation.
    Initialization can be done in one of the following ways:

    Args:
        1 (geometry_msgs/Pose): initialize from geometry_msgs.
        2 (Point, float): Second argument is the poses's orientation angle in radians.
        3 (Point, pyquaternion.Quaternion): Point and quaternion.


    Attributes:
        orientation (pyquaternion.Quaternion)
    """

    def __init__(self, *args):
        """Pose initialization."""

        with suppress(Exception):
            args = (args[0], Quaternion(*args[1]))

        with suppress(Exception):
            if isinstance(args[1], numbers.Number):
                args = (args[0], Quaternion(axis=[0, 0, 1], radians=args[1]))

        # Attempt default init
        with suppress(IndexError, NotImplementedError, TypeError):
            if type(args[1]) == Quaternion:
                self.orientation = args[1]
                super(Pose, self).__init__(args[0])
                return

        # Try to initialize geometry pose
        with suppress(Exception):
            # Call this function with values extracted
            t = Point(args[0].position)
            g_quaternion = args[0].orientation
            q = Quaternion(g_quaternion.w, g_quaternion.x, g_quaternion.y, g_quaternion.z)
            self.__init__(t, q)
            return

        # Try to initialize geometry transform
        with suppress(Exception):
            # Call this function with values extracted
            t = Point(args[0].translation)
            g_quaternion = args[0].rotation
            q = Quaternion(g_quaternion.w, g_quaternion.x, g_quaternion.y, g_quaternion.z)
            self.__init__(t, q)
            return

        with suppress(Exception):
            # Try to initialize with two points translation+orientation
            t = args[0]
            orientation_vec = args[1].to_numpy()
            angle = (-1 if orientation_vec[1] < 0 else 1) * math.acos(
                np.dot([1, 0, 0], orientation_vec) / np.linalg.norm(orientation_vec)
            )

            self.__init__(t, angle)
            return

        # None of the initializations worked
        raise NotImplementedError(f"Point initialization not implemented for {type(args[0])}")

    def get_angle(self) -> float:
        """Angle of orientation.

        Returns:
            The angle that a vector is rotated, when this orientation is applied as a transformation."""
        rot = self.orientation.normalised.rotate([1, 0, 0])

        assert len(rot) == 3
        assert abs(rot[0]) <= 1

        sign = -1 if rot[1] < 0 else 1

        try:
            return sign * math.acos(rot[0])
        except Exception as e:
            print(e)
            return 0

    def to_geometry_msg(self) -> geometry_msgs.Pose:
        """To ROS geometry_msg.

        Returns:
            This pose as a geometry_msgs/Pose.
        """
        point = super(Pose, self).to_geometry_msg()
        orientation = geometry_msgs.Quaternion(
            self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w
        )

        pose = geometry_msgs.Pose()
        pose.position = point
        pose.orientation = orientation

        return pose

    def __rmul__(self, tf: "Transform") -> "Pose":
        """Apply transformation.

        Args:
            tf (Transform): Transformation to apply.

        Returns:
            Pose transformed by tf.
        """
        with suppress(NotImplementedError, AttributeError):
            return Pose(tf * Vector(self), self.get_angle() + tf.get_angle())

        return NotImplemented

    def __eq__(self, pose):
        return pose.orientation == self.orientation and self.to_numpy().all() == pose.to_numpy().all()
