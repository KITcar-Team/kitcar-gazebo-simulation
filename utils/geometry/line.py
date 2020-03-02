# -*- coding: utf-8 -*-
"""Line."""

import shapely.geometry  # Base class
import shapely.affinity as affinity

import numpy as np
import geometry_msgs.msg as geometry_msgs
from road_generation import schema

from geometry.point import Point
from geometry.vector import Vector
from geometry.transform import Transform

from contextlib import suppress

from . import export

__copyright__ = "KITcar"


@export
class Line(shapely.geometry.linestring.LineString):
    """List of points as a Line class inheriting from shapely's LineString class.

    Inheriting from shapely enables to use their powerful operations in combination with other objects,
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
        raise NotImplementedError(f"Line initialization not implemented for {type(args[0])}")

    def get_points(self) -> [Point]:
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
        points = self.get_points()
        new_line = []
        for i in range(len(points)):
            if i != len(points) - 1:
                p1 = points[i]
                v = Vector(points[i + 1]) - Vector(points[i])
            else:
                p1 = points[i]

            if side == "left":
                v_orth = Vector((v.y * -1, v.x))
            elif side == "right":
                v_orth = Vector((v.y, v.x * -1))

            v_scaled = (offset / v_orth.norm) * v_orth
            p = p1 + v_scaled

            new_line.append(p)
        return Line(new_line)

    def to_schema_boundary(self) -> schema.boundary:
        """To schema.boundary.

        Export line as the boundary of a schema lanelet. E.g. the left boundery of the right lanelet (= middle line).

        Returns:
            Line as schema.boundary
        """
        boundary = schema.boundary()
        boundary.point = [p.to_schema() for p in self.get_points()]
        return boundary

    def to_geometry_msgs(self) -> [geometry_msgs.Point]:
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
            Lines concatenated behind another. """
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

    def __eq__(self, line):
        return self.get_points() == line.get_points()
