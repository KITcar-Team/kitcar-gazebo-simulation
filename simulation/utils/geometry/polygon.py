"""Polygon"""

__copyright__ = "KITcar"

import shapely.geometry  # Base class
import shapely.affinity as affinity

import numpy as np
import geometry_msgs.msg as geometry_msgs

from simulation.utils.geometry.point import Point
from simulation.utils.geometry.transform import Transform

from contextlib import suppress

from typing import List


class Polygon(shapely.geometry.polygon.Polygon):
    """Polygon class inheriting from shapely's Polygon class.

    Inheriting from shapely enables to use their powerful operations in combination \
    with other objects,
    e.g. polygon intersections.

    Initialization can be done in one of the following ways.

    Args:
        1 ([Point]): List of points or anything that can be initialized as a point,
                     e.g. Vector, geometry_msgs.Point,np.array
        2 (geometry_msgs.Polygon)
        3 ((Line,Line)): Two lines which are interpreted as the left \
                         and right boundary of the polygon, e.g. a road lane
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
        raise NotImplementedError(
            f"Polygon initialization not implemented for {type(args[0])}"
        )

    def get_points(self) -> List[Point]:
        """Points of polygon.

        Returns:
            list of points on the polygon.
        """
        return [Point(x, y, z) for x, y, z in self.exterior.coords]

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

    def __eq__(self, polygon: "Polygon"):
        """Compare two polygons using shapely's almost_equals.

        Also allow the points to be provided in the reversed order.
        """
        return self.almost_equals(polygon) or self.almost_equals(
            Polygon(reversed(polygon.get_points()))
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.get_points()})"
