"""
Basic transformation class which can be applied to other objects to modify them 

Author: Konstantin Ditschuneit
Date: 27.12.2019
"""

#### Compatible formats ####
import geometry_msgs.msg as geometry_msgs
from base.vector import Vector
from pyquaternion import Quaternion

import numbers
import math
import numpy as np


class Transform(Vector):

    def __init__(self, *args):
        """ Initialize pose from @args 

            @args can be one of the following types:
                - geometry_msgs/Transformation
                - (Vector, float)
                - (Vector, pyquaternion.Quaternion)
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
                self.rotation = args[1]
                super(Transform, self).__init__(args[0])
                return
        except:
            pass

        # Try to initialize geometry pose
        try:
            # Call this function with values extracted
            t = Vector(args[0].position)
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
            t = Vector(args[0].translation)
            g_quaternion = args[0].rotation
            q = Quaternion(g_quaternion.w, g_quaternion.x,
                           g_quaternion.y, g_quaternion.z)
            self.__init__(t, q)
            return
        except:
            pass

        try:
            # Try to initialize with two vectors translation+rotation
            t = args[0]
            rotation_vec = args[1].to_numpy()
            angle = (-1 if rotation_vec[1] < 0 else 1) * math.acos(
                np.dot([1, 0, 0], rotation_vec)/np.linalg.norm(rotation_vec))

            self.__init__(t, angle)
            return
        except:
            pass
        # None of the initializations worked
        raise NotImplementedError(
            f"Transform initialization not implemented for {type(args[0])}")

    def get_angle(self):
        rot = self.rotation.normalised.rotate([1, 0, 0])

        assert(len(rot) == 3)
        assert(abs(rot[0]) <= 1)

        sign = -1 if rot[1] < 0 else 1

        try:
            return sign * math.acos(rot[0])
        except Exception as e:
            print(e)
            return 0

    def to_geometry_msg(self):
        """ Convert vector to geometry_msgs/Point """
        vector = super(Transform, self).to_geometry_msg()
        rotation = geometry_msgs.Quaternion(
            self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w)

        tf = geometry_msgs.Transform()
        tf.translation = vector
        tf.rotation = rotation

        return tf

    def __mul__(self, tf):

        try:
            return Transform(self + Vector(tf).rotated(self.get_angle()), self.get_angle() + tf.get_angle())
        except:
            pass

        return NotImplemented

    def __eq__(self, tf):
        return tf.rotation.normalised == self.rotation.normalised and self.to_numpy().all() == tf.to_numpy().all()
