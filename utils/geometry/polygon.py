# -*- coding: utf-8 -*-
"""Polygon"""

import shapely.geometry  # Base class
import shapely.affinity as affinity

import numpy as np
import geometry_msgs.msg as geometry_msgs
from road_generation import schema

from geometry.point import Point
from geometry.line import Line
from geometry.transform import Transform

from contextlib import suppress

__author__ = "Konstantin Ditschuneit"
__copyright__ = "KITcar"


class Polygon(shapely.geometry.polygon.Polygon):
    """Polygon class inheriting from shapely's Polygon class.

    Inheriting from shapely enables to use their powerful operations in combination with other objects,
    e.g. polygon intersections.

    Initialization can be done in one of the following ways.

    Args:
        1 ([Point]): List of points or anything that can be initialized as a point,
                     e.g. Vector, geometry_msgs.Point,np.array
        2 (geometry_msgs.Polygon)
        3 ((Line,Line)): Two lines which are interpreted as the left and right boundary of the polygon,
                         e.g. a road lane
    """

    def __init__(self, *args):
        """Polygon initialization."""
        # Create points out of arguments:
        with suppress(NotImplementedError, IndexError, TypeError):
            args = ([Point(p) for p in args[0]], 0)  # Make tuple

        # Try to initialize directly
        with suppress(Exception):
            super(Polygon, self).__init__(*args)
            return

        # Try to initialize from list of points
        with suppress(Exception):
            args = [[p.x, p.y, p.z] for p in args[0]]
            super(Polygon, self).__init__(args)
            return

        # Try to initialize from geometry_msgs/Polygon
        with suppress(Exception):
            super(Polygon, self).__init__([[p.x, p.y, p.z] for p in args[0].points])
            return

        # Try to initialize from two lines
        with suppress(AttributeError):
            points = args[1].get_points()
            points.extend(reversed(args[0].get_points()))

            self.__init__(points)
            return

        # None of the initializations worked
        raise NotImplementedError(f"Polygon initialization not implemented for {type(args[0])}")

    def get_points(self) -> [Point]:
        """Points of polygon.

        Returns:
            list of points on the polygon.
        """
        return [Point(x, y, z) for x, y, z in self.exterior.coords]

    def to_schema_lanelet(self, split_idx: int = 0) -> schema.lanelet:
        """To schema lanelet.

        This is done by splitting the polygon in two parts.
        The first number of points (split_idx) are considered to
        be on the right side. The returned lanelet has no line markings.

        Args:
            split_idx (int): Points in polygon before this index are considered as the right boundary
            of the resulting lanelet, the other points as the left boundary.

        Returns:
            Schema lanelet without lane markings from polygon
        """
        if split_idx == 0:
            split_idx = int((len(self.exterior.coords) - 1) / 2)

        right_line = Line(self.exterior.coords[:split_idx])
        left_line = Line(reversed(self.exterior.coords[split_idx:-1]))

        # Create lanelet
        lanelet = schema.lanelet()
        lanelet.rightBoundary = right_line.to_schema_boundary()
        lanelet.leftBoundary = left_line.to_schema_boundary()

        return lanelet

    def to_geometry_msg(self):
        """To ROS geometry_msg.

        Returns:
            This polygon as a geometry_msgs.Polygon.
        """
        msg = geometry_msgs.Polygon()
        msg.points = [geometry_msgs.Point32(*p.to_numpy()) for p in self.get_points()]
        return msg

    def to_numpy(self):
        """To numpy array.

        Returns:
            Polygon as a numpy array of np.arrays.
        """
        return np.array([p.to_numpy() for p in self.get_points()])

    def __rmul__(self, tf: Transform):
        """ Transform this polygon.

        Args:
            tf (Transform): Transformation to apply

        Rotate the polygon tf.rotation around (0,0,0) and translate by tf.xyz

        Returns:
            Transformed polygon.
        """
        if not type(tf) is Transform:
            return NotImplemented

        transformed = affinity.rotate(self, tf.get_angle(), use_radians=True, origin=[0, 0])
        transformed = affinity.translate(transformed, tf.x, tf.y, tf.z)

        return self.__class__(transformed.exterior.coords)

    def __eq__(self, polygon):
        eq = self.get_points() == polygon.get_points()
        rev_eq = self.get_points() == list(reversed(polygon.get_points()))
        return eq or rev_eq
