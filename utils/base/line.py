"""
Basic line class which is compatible with all needed formats

Author: Konstantin Ditschuneit
Date: 15.11.2019
"""

import shapely.geometry  # Base class
import shapely.affinity as affinity

import numpy as np
import geometry_msgs.msg as geometry_msgs
from road_generation import schema

from base.point import Point
from base.vector import Vector
from base.transform import Transform

class Line(shapely.geometry.linestring.LineString):

    def __init__(self, *args):
        """ Initialize line from different formats."""

        if len(args) == 0:
            args = ([], None)

        # Catch missing z coordinate by converting to point
        try:
            args = ([Point(arg) for arg in args[0]], None)
        except:
            pass

        # Try to initialize from list of Point or geometry_msgs/Point
        try:
            super(Line, self).__init__([[p.x, p.y, p.z] for p in args[0]])
            return
        except:
            pass

        # None of the initializations worked
        raise NotImplementedError(
            f"Line initialization not implemented for {type(args[0])}")

    def get_points(self):
        """
        Returns list of points that shape the polygon
        """
        return [Point(x, y, z) for x, y, z in self.coords]

    def parallel_offset(self, offset, side):
        points = self.get_points()
        new_line = []
        for i in range(len(points)):
            if i != len(points) - 1:
                p1 = points[i]
                v = Vector(points[i + 1]) - Vector(points[i])
            else:
                p1 = points[i]

            if side == 'left':
                v_orth = Vector((v.y * -1, v.x))
            elif side == 'right':
                v_orth = Vector((v.y, v.x * -1))

            v_scaled = (offset / v_orth.norm) * v_orth
            p = p1 + v_scaled

            new_line.append(p)
        return Line(new_line)

    def to_schema_boundary(self):
        """
        Conversion to a schema.boundary
        """
        boundary = schema.boundary()
        boundary.point = [p.to_schema() for p in self.get_points()]
        return boundary

    def to_geometry_msgs(self):
        """
        Convert to [geometry_msgs/Point]
        """
        return [p.to_geometry_msg() for p in self.get_points()]

    def to_numpy(self):
        return np.array([p.to_numpy() for p in self.get_points()])

    def __add__(self, line):
        """ Addition of two lines just adds them behind another. """
        coords = list(self._get_coords())
        coords.extend(line._get_coords())

        return Line(coords)

    def __rmul__(self, tf):
        """ Transform this line by @tf:Transform.

        Rotate the line @tf.rotation around (0,0,0) and translate by @tf.xyz
        """
        if not type(tf) is Transform:
            return NotImplemented

        transformed = affinity.rotate(
            self, tf.get_angle(), use_radians=True, origin=[0, 0])
        transformed = affinity.translate(transformed, tf.x, tf.y, tf.z)

        return Line(transformed.coords)

    def __eq__(self, line):
        return self.get_points() == line.get_points()
