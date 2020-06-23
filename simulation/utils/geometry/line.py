"""Line."""

__copyright__ = "KITcar"


import shapely.geometry  # Base class
import shapely.affinity as affinity

import numpy as np
import geometry_msgs.msg as geometry_msgs

from simulation.utils.geometry.point import Point
from simulation.utils.geometry.vector import Vector
from simulation.utils.geometry.transform import Transform
from simulation.utils.geometry.pose import Pose

from contextlib import suppress

from typing import List, Callable

#
APPROXIMATION_DISTANCE = 0.00005
CURVATURE_APPROX_DISTANCE = 0.04
SIMPLIFICATION_TOLERANCE = 0.001


def ensure_valid_arc_length(*, approx_distance=APPROXIMATION_DISTANCE) -> Callable:
    """Check if an arc length is on the line and can be used for approximation.

    If the arc_length is too close to the end points of the line, \
    it is moved further away from the edges.

    Args:
        approx_distance(float): Approximation step length to be used \
                in further calculations.
                Arc length will be at least that far away from the end of the line.
    """

    def wrapper(func):
        def decorator(self, arc_length: float):

            # Ensure that arc_length is not too close to the end points of the road.
            if not (arc_length >= 0 and arc_length <= self.length):
                raise ValueError(
                    "The provided arc length is less than 0 \
                            or greater than the line's length."
                )
            elif self.length == 0:
                raise ValueError("The line is too short.")

            arc_length = max(arc_length, approx_distance)
            arc_length = min(arc_length, self.length - approx_distance)

            return func(self, arc_length=arc_length)

        decorator.__doc__ = func.__doc__
        decorator.__annotations__ = func.__annotations__

        return decorator

    return wrapper


class Line(shapely.geometry.linestring.LineString):
    """List of points as a Line class inheriting from shapely's LineString class.

    Inheriting from shapely enables to use their powerful operations in combination \
    with other objects,
    e.g. polygon intersections.

    Initialization can be done in one of the following ways.

    Args:
        1 ([Point]): List of points or anything that can be initialized as a point,
                     e.g. Vector, geometry_msgs.Point,np.array)
        2 ([]): Empty list creates an empty line.

    """

    def __init__(self, *args):
        """Line initialization."""
        if len(args) == 0:
            args = ([], None)

        # Catch missing z coordinate by converting to point
        with suppress(NotImplementedError, IndexError):
            args = ([Point(arg) for arg in args[0]], None)

        # Try to initialize from list of Point or geometry_msgs/Point
        with suppress(NotImplementedError, AttributeError):
            super(Line, self).__init__([[p.x, p.y, p.z] for p in args[0]])
            return

        # None of the initializations worked
        raise NotImplementedError(
            f"Line initialization not implemented for {type(args[0])}"
        )

    def get_points(self) -> List[Point]:
        """Points of line.

        Returns:
            list of points on the line.
        Rotate the line tf.rotation around (0,0,0) and translate by tf.xyz
        """
        return [Point(x, y, z) for x, y, z in self.coords]

    def parallel_offset(self, offset: float, side: str) -> "Line":
        """Shift line.

        Args:
            offset (float): distance to shift
            side (str): either `left` or `right` shift

        Returns:
            Line shifted by an offset into the left or right direction.
        """
        assert (
            side == "right" or side == "left"
        ), "Parallel offset is only possible to the right or left!"
        coords = super().parallel_offset(offset, side).coords
        if side == "right":
            # Because shapely orders right hand offset lines in reverse
            coords = reversed(coords)
        return Line(coords)

    @ensure_valid_arc_length()
    def interpolate_direction(self, *, arc_length: float) -> Vector:
        """Interpolate the direction of the line as a vector.

        Approximate by calculating difference vector of a point slightly further
        and a point slightly before along the line.

        Args:
            arc_length (float): Length along the line starting from the first point

        Raises:
            ValueError: If the arc_length is <0 or more than the length of the line.

        Returns:
            Corresponding direction as a normalised vector.
        """
        n = Vector(self.interpolate(arc_length + APPROXIMATION_DISTANCE))
        p = Vector(self.interpolate(arc_length - APPROXIMATION_DISTANCE))

        d = n - p

        return 1 / abs(d) * d

    @ensure_valid_arc_length(approx_distance=CURVATURE_APPROX_DISTANCE)
    def interpolate_curvature(self, *, arc_length: float) -> float:
        """Interpolate the curvature at a given arc_length.

        The curvature is approximated by calculating the Menger curvature as defined \
        and described here:
        https://en.wikipedia.org/wiki/Menger_curvature#Definition

        Args:
            arc_length (float): Length along the line starting from the first point

        Raises:
            ValueError: If the arc_length is <0 or more than the length of the line.

        Returns:
            Corresponding curvature.
        """
        p = Vector(
            self.interpolate(arc_length - CURVATURE_APPROX_DISTANCE)
        )  # Previous point
        c = Vector(self.interpolate(arc_length))  # Point at current arc_length
        n = Vector(self.interpolate(arc_length + CURVATURE_APPROX_DISTANCE))  # Next point

        # Area of the triangle spanned by p, c, and n.
        # The triangle's area can be computed by the cross product of the vectors.
        cross = (n - c).cross(p - c)

        sign = 1 - 2 * (
            cross.z < 0
        )  # The z-coord sign determines whether the curvature is positive or negative

        return sign * 2 * abs(cross) / (abs(p - c) * abs(n - c) * abs(p - n))

    @ensure_valid_arc_length(approx_distance=0)
    def interpolate_pose(self, *, arc_length: float) -> Pose:
        """Interpolate the pose a model travelling along this line has.

        Args:
            arc_length (float): Length along the line starting from the first point

        Raises:
            ValueError: If the arc_length is <0 or more than the length of the line.

        Returns:
            Corresponding pose.
        """
        point = self.interpolate(arc_length)
        orientation = self.interpolate_direction(arc_length=arc_length)

        return Pose(Point(point), orientation)

    def to_geometry_msgs(self) -> List[geometry_msgs.Point]:
        """To ROS geometry_msgs.

        Returns:
            This line as a list of geometry_msgs/Point.
        """
        return [p.to_geometry_msg() for p in self.get_points()]

    def to_numpy(self) -> np.ndarray:
        """To numpy array.

        Returns:
            Line as a numpy array of np.arrays.
        """
        return np.array([p.to_numpy() for p in self.get_points()])

    def __add__(self, line: "Line"):
        """Concatenate lines.

        Returns:
            Lines concatenated behind another.
        """
        coords = list(self._get_coords())
        coords.extend(line._get_coords())

        return self.__class__(coords)

    def __rmul__(self, tf: Transform):
        """ Transform this line.

        Args:
            tf (Transform): Transformation to apply

        Rotate the line tf.rotation around (0,0,0) and translate by tf.xyz

        Returns:
            Transformed line.
        """
        if not type(tf) is Transform:
            return NotImplemented

        transformed = affinity.rotate(self, tf.get_angle(), use_radians=True, origin=[0, 0])
        transformed = affinity.translate(transformed, tf.x, tf.y, tf.z)

        return self.__class__(transformed.coords)

    def __eq__(self, line: "Line") -> bool:
        if not self.__class__ == line.__class__:
            return NotImplemented

        # simplify lines (1 mm tolerance) before checking equality
        return self.simplify(tolerance=SIMPLIFICATION_TOLERANCE).almost_equals(
            line.simplify(tolerance=SIMPLIFICATION_TOLERANCE)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.get_points()})"
