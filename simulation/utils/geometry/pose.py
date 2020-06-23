"""Pose."""

__copyright__ = "KITcar"

# Compatible formats
import geometry_msgs.msg as geometry_msgs
from simulation.utils.geometry.vector import Vector
from simulation.utils.geometry.point import Point
from simulation.utils.geometry.transform import Transform
from pyquaternion import Quaternion

import numbers
import math

from contextlib import suppress


class Pose(Point):
    """Pose class consisting of a position and an orientation.

    A Pose object can be used to describe the state of an object with a position \
    and an orientation.
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
            orientation_vec = Vector(args[1])
            angle = (-1 if orientation_vec.y < 0 else 1) * math.acos(
                (Vector(1, 0, 0) * orientation_vec) / abs(orientation_vec)
            )

            self.__init__(t, angle)
            return

        # None of the initializations worked
        raise NotImplementedError(
            f"{self.__class__} initialization not implemented for {type(args[0])}"
        )

    def get_angle(self) -> float:
        """Angle of orientation.

        Returns:
            The angle that a vector is rotated, when this transformation is applied.
        """
        # Project the rotation axis onto the z axis to get the amount of the rotation \
        # that is in z direction!
        # Also the quaternions rotation axis is sometimes (0,0,-1) at which point \
        # the angles flip their sign,
        # taking the scalar product of the axis and z fixes that as well
        return Vector(self.orientation.axis) * Vector(0, 0, 1) * self.orientation.radians

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

    def __rmul__(self, tf: "Transform"):
        """Apply transformation.

        Args:
            tf (Transform): Transformation to apply.

        Returns:
            Pose transformed by tf.
        """
        with suppress(NotImplementedError, AttributeError):
            return self.__class__(tf * Vector(self), self.get_angle() + tf.get_angle())

        return NotImplemented

    def __eq__(self, pose) -> bool:
        TOLERANCE = 1e-8  # Radian!
        if self.__class__ != pose.__class__:
            return NotImplemented
        return (
            self.get_angle() - pose.get_angle()
        ) < TOLERANCE and self.to_numpy().all() == pose.to_numpy().all()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(position={self.x, self.y, self.z},"
            f"orientation= {round(math.degrees(self.get_angle()),4)} degrees)"
        )

    def __hash__(self):
        return NotImplemented
