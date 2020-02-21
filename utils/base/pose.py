"""
Basic point class which is compatible with all needed formats

Author: Konstantin Ditschuneit
Date: 15.11.2019
"""

#### Compatible formats ####
import geometry_msgs.msg as geometry_msgs
from base.vector import Vector
from base.point import Point
from base.transform import Transform
from pyquaternion import Quaternion

import numbers
import math
import numpy as np

from contextlib import suppress

class Pose(Point):

    def __init__(self, *args):
        """ Initialize pose from @args 

            @args can be one of the following types:
                - geometry_msgs/Pose
                - (Point, float)
                - (Point, pyquaternion.Quaternion)
        """
        try:
            args = (args[0], Quaternion(*args[1]))
        except:
            pass

        try:
            if isinstance(args[1], numbers.Number):
                args = (args[0], Quaternion(axis=[0, 0, 1], radians=args[1]))
            pass
        except:
            pass

      # Attempt default init
        try:

            if type(args[1]) == Quaternion:
                self.orientation = args[1]
                super(Pose, self).__init__(args[0])
                return
        except:
            pass

        # Try to initialize geometry pose
        try:
            # Call this function with values extracted
            t = Point(args[0].position)
            g_quaternion = args[0].orientation
            q = Quaternion(g_quaternion.w, g_quaternion.x,
                           g_quaternion.y, g_quaternion.z)
            self.__init__(t, q)
            return
        except:
            pass
        # Try to initialize geometry transform
        try:
            # Call this function with values extracted
            t = Point(args[0].translation)
            g_quaternion = args[0].rotation
            q = Quaternion(g_quaternion.w, g_quaternion.x,
                           g_quaternion.y, g_quaternion.z)
            self.__init__(t, q)
            return
        except:
            pass

        try:
            # Try to initialize with two points translation+orientation
            t = args[0]
            orientation_vec = args[1].to_numpy()
            angle = (-1 if orientation_vec[1] < 0 else 1) * math.acos(
                np.dot([1, 0, 0], orientation_vec)/np.linalg.norm(orientation_vec))

            self.__init__(t, angle)
            return
        except:
            pass
        # None of the initializations worked
        raise NotImplementedError(
            f"Point initialization not implemented for {type(args[0])}")

    def get_angle(self):
        rot = self.orientation.normalised.rotate([1, 0, 0])

        assert(len(rot) == 3)
        assert(abs(rot[0]) <= 1)

        sign = -1 if rot[1] < 0 else 1

        try:
            return sign * math.acos(rot[0])
        except Exception as e:
            print(e)
            return 0

    def to_geometry_msg(self):
        """ Convert point to geometry_msgs/Point """
        point = super(Pose, self).to_geometry_msg()
        orientation = geometry_msgs.Quaternion(
            self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w)

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
            return Pose(tf*Vector(self), self.get_angle() + tf.get_angle())

        return NotImplemented

    def __eq__(self, pose):
        return pose.orientation == self.orientation and self.to_numpy().all() == pose.to_numpy().all()
