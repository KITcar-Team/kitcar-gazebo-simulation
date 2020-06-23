"""Transformation"""

__copyright__ = "KITcar"

# Compatible formats
import geometry_msgs.msg as geometry_msgs
from simulation.utils.geometry.vector import Vector
from pyquaternion import Quaternion

import numbers
import math
import numpy as np

from contextlib import suppress


class Transform(Vector):
    """Transformation class consisting of a translation and a rotation.

    A Transform object can be used to easily interchange between multiple coordinate systems
    (which can be transformed in one another via a rotation and a translation.

    Initialization can be done in one of the following ways:

    Args:
        1 (geometry_msgs/Transformation): initialize from geometry_msgs.
        2 (Vector, float): Second argument is the transformation's rotation angle in radians.
        3 (Vector, pyquaternion.Quaternion): Vector and quaternion.


    Attributes:
        rotation (pyquaternion.Quaternion)
    """

    def __init__(self, *args):
        """Transform initialization."""

        # Attempt initialization from Vector like and Quaternion like objects
        with suppress(Exception):
            args = (args[0], Quaternion(*args[1]))

        with suppress(Exception):
            if isinstance(args[1], numbers.Number):
                args = (args[0], Quaternion(axis=[0, 0, 1], radians=args[1]))
            pass

        # Attempt default init
        with suppress(IndexError, NotImplementedError, TypeError):
            if type(args[1]) == Quaternion:
                self.rotation = args[1]
                super(Transform, self).__init__(args[0])
                return

        # Try to initialize geometry pose
        with suppress(Exception):
            # Call this function with values extracted
            t = Vector(args[0].position)
            g_quaternion = args[0].orientation
            q = Quaternion(g_quaternion.w, g_quaternion.x, g_quaternion.y, g_quaternion.z)
            self.__init__(t, q)
            return

        # Try to initialize geometry transform
        with suppress(Exception):
            # Call this function with values extracted
            t = Vector(args[0].translation)
            g_quaternion = args[0].rotation
            q = Quaternion(g_quaternion.w, g_quaternion.x, g_quaternion.y, g_quaternion.z)
            self.__init__(t, q)
            return

        with suppress(Exception):
            # Try to initialize with two vectors translation+rotation
            t = args[0]
            rotation_vec = args[1].to_numpy()
            angle = (-1 if rotation_vec[1] < 0 else 1) * math.acos(
                np.dot([1, 0, 0], rotation_vec) / np.linalg.norm(rotation_vec)
            )

            self.__init__(t, angle)
            return

        # None of the initializations worked
        raise NotImplementedError(
            f"Transform initialization not implemented for {type(args[0])}"
        )

    @property
    def inverse(self) -> "Transform":
        """Inverse transformation."""
        return Transform(
            -1 * Vector(self).rotated(-self.get_angle()), -1 * self.get_angle()
        )

    def get_angle(self) -> float:
        """Angle of rotation.

        Returns:
            The angle that a vector is rotated, when this transformation is applied."""

        # Project the rotation axis onto the z axis to get the amount of the rotation \
        # that is in z direction!
        # Also the quaternions rotation axis is sometimes (0,0,-1) at which point \
        # the angles flip their sign,
        # taking the scalar product of the axis and z fixes that as well
        return Vector(self.rotation.axis) * Vector(0, 0, 1) * self.rotation.radians

    def to_geometry_msg(self) -> geometry_msgs.Transform:
        """Convert transform to ROS geometry_msg.

        Returns:
            This transformation as a geometry_msgs/Transform.
        """
        vector = super(Transform, self).to_geometry_msg()
        rotation = geometry_msgs.Quaternion(
            self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w
        )

        tf = geometry_msgs.Transform()
        tf.translation = vector
        tf.rotation = rotation

        return tf

    def __mul__(self, tf: "Transform") -> "Transform":
        """Multiplication of transformations.

        The product has to be understood as a single transformation consisting of
        the right hand transformation applied first and then the left hand transformation.

        Example:
            Easily modify a vector multiple times:

            :math:`(\\text{Tf}_1*\\text{Tf}_2)*\\vec{v} = \\text{Tf}_1*( \\text{Tf}_2*\\vec{v})`

        Returns:
            The product transformation.
        """
        if tf.__class__ == self.__class__:
            return self.__class__(
                Vector(self) + Vector(tf).rotated(self.get_angle()),
                self.get_angle() + tf.get_angle(),
            )

        return NotImplemented

    def __eq__(self, tf) -> bool:
        if self.__class__ != tf.__class__:
            return NotImplemented
        return tf.rotation.normalised == self.rotation.normalised and Vector(
            self
        ) == Vector(tf)

    def __repr__(self) -> str:
        return (
            f"Transform(translation={self.x, self.y, self.z},"
            f"rotation={round(math.degrees(self.get_angle()),4)} degrees)"
        )

    def __hash__(self):
        return NotImplemented
