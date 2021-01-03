"""Vector."""

import math
import numbers
from contextlib import suppress
from typing import Union
from warnings import warn

import geometry_msgs.msg as geometry_msgs
import numpy as np
import shapely.geometry  # Base class
from pyquaternion import Quaternion

from .frame import validate_and_maintain_frames


class Vector(shapely.geometry.point.Point):
    """Implementation of the mathematical vector object.

    Inheriting from shapely enables to use their powerful operations in combination with
    other objects, e.g. lines, polygons.

    Initialization can be done in one of the following ways.

    Args:
        [x,y,z] (np.array)
        geometry_msgs/Vector
        (x,y*,z*) (float)

    Keyword Args:
        r (float): length of vector
        phi (float): angle between vector and x-y-plane

    A vector is always initialized with 3 coordinates.
    If there's no third coordinate provided, z:=0.
    """

    def __init__(self, *args, **kwargs):
        """Vector initialization."""

        # Due to recursive calling of the init function, the frame should be set
        # in the first call within the recursion only.
        if not hasattr(self, "_frame"):
            self._frame = kwargs.get("frame", None)

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

        with suppress(AttributeError):
            # Shapely point without z!
            if not args[0].has_z:
                args = [args[0].x, args[0].y, 0]

        # Attempt default init
        with suppress(Exception):
            super(Vector, self).__init__(*args)
            return

        if isinstance(args[0], (geometry_msgs.Vector3, geometry_msgs.Point)):
            warn(
                f"Initializing {self.__class__} with {args[0].__class__}"
                "directly is deprecated."
                f"Use {self.__class__.from_geometry_msg} instead.",
                DeprecationWarning,
            )

        # Try to initialize geometry vector
        with suppress(AttributeError):
            # Call this function with values extracted
            self.__init__(args[0].x, args[0].y, args[0].z)
            return

        # None of the initializations worked
        raise NotImplementedError(
            f"{type(self).__name__} initialization not implemented for {type(args[0])}"
        )

    @classmethod
    def from_geometry_msg(cls, geometry_msg: geometry_msgs.Vector3):
        """Initialize from ROS geometry_msg."""
        return cls(geometry_msg.x, geometry_msg.y, geometry_msg.z)

    def to_geometry_msg(self) -> geometry_msgs.Vector3:
        """To ROS geometry_msg.

        Returns:
            This vector as a geometry_msgs/Vector3
        """
        return geometry_msgs.Vector3(x=self.x, y=self.y, z=self.z)

    def to_numpy(self) -> np.ndarray:
        """To numpy array.

        Returns:
            Vector as a numpy array.
        """
        return np.array([self.x, self.y, self.z])

    @validate_and_maintain_frames
    def rotated(self, arg: Union[float, Quaternion]):
        """This vector rotated around [0,0,0].

        Args:
            arg: Rotation angle in radian or quaternion to rotate by.

        Returns:
            Rotated vector.
        """
        if isinstance(arg, Quaternion):
            return self.__class__(arg.rotate(self.to_numpy()))
        else:
            c = math.cos(arg)
            s = math.sin(arg)
            # Matrix multiplication
            return self.__class__(c * self.x - s * self.y, s * self.x + c * self.y, self.z)

    @validate_and_maintain_frames
    def cross(self, vec: "Vector") -> "Vector":
        """Cross product with other vector.

        Args:
            vec(Vector): Second vector.

        Returns:
            Resulting vector.
        """

        x = self.y * vec.z - self.z * vec.y
        y = self.z * vec.x - self.x * vec.z
        z = self.x * vec.y - self.y * vec.x

        return Vector(x, y, z)

    @property
    def argument(self) -> float:
        """float: Return the argument of the vector in radian."""
        return math.atan2(self.y, self.x)

    @validate_and_maintain_frames
    def __sub__(self, vec):
        return self.__class__(self.x - vec.x, self.y - vec.y, self.z - vec.z)

    def __abs__(self):
        """Eucledian norm of the vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    @validate_and_maintain_frames
    def __add__(self, vec):
        return self.__class__(self.x + vec.x, self.y + vec.y, self.z + vec.z)

    @validate_and_maintain_frames
    def __mul__(self, vec: "Vector") -> float:
        """Scalar product.

        Returns:
            :math:`\\vec{vec} \\cdot \\vec{v}`
        """
        if isinstance(vec, self.__class__):
            return vec.x * self.x + vec.y * self.y + vec.z * self.z

        return NotImplemented

    @validate_and_maintain_frames
    def __rmul__(self, c):
        """Scale or transform vector by c.

        Args:
            c: Scalar or Transform

        Returns:
            :math:`c \\cdot \\vec{v}`
        """

        # If c is a transform
        with suppress(AttributeError):
            return self.rotated(c.rotation) + self.__class__(c.translation)

        if isinstance(c, numbers.Number):
            return self.__class__(c * self.x, c * self.y, c * self.z)

        return NotImplemented

    @validate_and_maintain_frames
    def __eq__(self, vec) -> bool:
        if not self.__class__ == vec.__class__:
            return NotImplemented
        return self.almost_equals(vec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}"
            f"{tuple(round(val,8) for val in [self.x,self.y,self.z])}"
            + (f"(frame: {self._frame.name})" if self._frame is not None else "")
        )

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z, self._frame))
