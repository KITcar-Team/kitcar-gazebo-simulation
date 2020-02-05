"""
Basic polygon class which is compatible with all needed formats

Author: Konstantin Ditschuneit
Date: 15.11.2019
"""

import shapely.geometry  # Base class
import shapely.affinity as affinity

import numpy as np
import geometry_msgs.msg as geometry_msgs
from road_generation import schema

from base.point import Point
from base.line import Line
from base.transform import Transform


class Polygon(shapely.geometry.polygon.Polygon):
    def __init__(self, *args):
        """ Initialize polygon from @args
            @args can be one of the following types:
                - np.array
                - geometry_msgs/Polygon
                - Line,Line (interpreted as left and right side (eg. a lane)!)
        """
        # Create points out of arguments:
        try:
            args = ([Point(p) for p in args[0]], 0)  # Make tuple
        except:
            pass

        # Try to initialize directly
        try:
            super(Polygon, self).__init__(*args)
            return
        except:
            pass

        # Try to initialize from list of points
        try:
            args = [[p.x, p.y, p.z] for p in args[0]]
            super(Polygon, self).__init__(args)
            return
        except:
            pass

        # Try to initialize from geometry_msgs/Polygon
        try:
            super(Polygon, self).__init__(
                [[p.x, p.y, p.z] for p in args[0].points]
            )
            return
        except:
            pass

        # Try to initialize from two lines
        try:
            points = args[1].get_points()
            points.extend(reversed(args[0].get_points()))

            self.__init__(points)
            return
        except:
            pass

        # None of the initializations worked
        raise NotImplementedError(
            f"Polygon initialization not implemented for {type(args[0])}"
        )

    def get_points(self):
        """
        Returns list of points that shape the polygon
        """
        return [Point(x, y, z) for x, y, z in self.exterior.coords]

    def to_schema_lanelet(self, split_idx=0):
        """
        Conversion to a schema.lanelet. This is done by splitting the polygon
        in two parts. The first @split_idx- number of points are considered to
        be on the right side. The returned lanelet has no line markings.
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
        """
        Convert to geometry_msgs/Polygon
        """
        msg = geometry_msgs.Polygon()
        msg.points = [
            geometry_msgs.Point32(*p.to_numpy()) for p in self.get_points()
        ]
        return msg

    def to_numpy(self):
        return np.array([p.to_numpy() for p in self.get_points()])

    def __rmul__(self, tf):
        """ Transform this polygon by @tf:Transform.

        Rotate the polygon @tf.rotation around (0,0,0) and translate by @tf.xyz
        """
        if not type(tf) is Transform:
            return NotImplemented

        transformed = affinity.rotate(
            self, tf.get_angle(), use_radians=True, origin=[0, 0]
        )
        transformed = affinity.translate(transformed, tf.x, tf.y, tf.z)

        return Polygon(transformed.exterior.coords)

    def __eq__(self, polygon):
        eq = self.get_points() == polygon.get_points()
        rev_eq = self.get_points() == list(reversed(polygon.get_points()))
        return eq or rev_eq
