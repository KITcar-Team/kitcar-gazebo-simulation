# -*- coding: utf-8 -*-
"""Vector"""

import shapely.geometry  # Base class
import math  # For calculations

import numpy as np
import geometry_msgs.msg as geometry_msgs
import numbers

from contextlib import suppress

__copyright__ = "KITcar"


class Vector(shapely.geometry.point.Point):
    """Vector class which inherits from shapely's Point class and implements a vector in the mathematical sense.

    Inheriting from shapely enables to use their powerful operations in combination with other objects,
    e.g. lines, polygons.

    Initialization can be done in one of the following ways.

    Args:
        [x,y,z] (np.array)
        geometry_msgs/Vector
        (x,y*,z*) (float)

    Keyword Args:
        r (float): length of vector
        phi (float): angle between vector and x-y-plane

    A vector is always initialized with 3 coordinates. If there's no third coordinate provided, z:=0.
    """

    def __init__(self, *args, **kwargs):
        """Vector initialization."""

        if "r" in kwargs and "phi" in kwargs:
            # construct Vector from r, phi
            r = kwargs["r"]
            phi = kwargs["phi"]
            self.__init__(math.cos(phi) * r, math.sin(phi) * r)
            return

        # Try to add z component
        with suppress(TypeError):
            if len(args) == 2:
                args = (*args, 0)
            elif len(args[0]) == 2:
                args = (*args[0], 0)

        # Attempt default init
        with suppress(Exception):
            super(Vector, self).__init__(*args)
            return

        # Try to initialize geometry vector
        with suppress(AttributeError):
            # Call this function with values extracted
            self.__init__(args[0].x, args[0].y, args[0].z)
            return

        # None of the initializations worked
        raise NotImplementedError(f"{type(self).__name__} initialization not implemented for {type(args[0])}")

    def to_geometry_msg(self) -> geometry_msgs.Vector3:
        """To ROS geometry_msg.

        Returns:
            This vector as a geometry_msgs/Vector3 """
        return geometry_msgs.Vector3(x=self.x, y=self.y, z=self.z)

    def to_numpy(self) -> np.ndarray:
        """To numpy array.

        Returns:
            Vector as a numpy array. """
        return np.array([self.x, self.y, self.z])

    @property
    def norm(self) -> float:
        """Eucledian norm of the vector.

        Returns:
            The norm as float.

        """
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def rotated(self, angle: float):
        """This vector rotated around [0,0,0] in the x-y-plane.

        Args:
            angle (float): Rotation angle in radian.

        Returns:
            Rotated vector.

        """
        c = math.cos(angle)
        s = math.sin(angle)
        # Matrix multiplication
        return self.__class__(c * self.x - s * self.y, s * self.x + c * self.y, self.z)

    def __sub__(self, vec):
        """Subtracted by a another vector."""
        return self.__class__(self.x - vec.x, self.y - vec.y, self.z - vec.z)

    def __add__(self, vec):
        """Another vector added."""
        return self.__class__(self.x + vec.x, self.y + vec.y, self.z + vec.z)

    def __mul__(self, vec: "Vector") -> float:
        """Scalar product.

        Returns:
            :math:`\\vec{vec} \\cdot \\vec{v}`
            """
        if isinstance(vec, self.__class__):
            return vec.x * self.x + vec.y * self.y + vec.z * self.z

        return NotImplemented

    def __rmul__(self, c):
        """Scale vector by number c.

        Args:
            c (float, Transform): Scalar or Transform

        Returns:
            :math:`c \\cdot \\vec{v}`
        """

        # If c is a transform
        with suppress(AttributeError):
            return self.rotated(c.get_angle()) + self.__class__(c)

        if isinstance(c, numbers.Number):
            return self.__class__(c * self.x, c * self.y, c * self.z)

        return NotImplemented

    def __eq__(self, vec: "Vector") -> bool:
        """Decide if vectors are equal."""
        TOLERANCE = 1e-8
        return type(vec) == type(self) and (Vector(vec) - Vector(self)).norm < TOLERANCE
