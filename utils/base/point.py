"""
Basic point class which is compatible with all needed formats
"""

# Compatible formats
import geometry_msgs.msg as geometry_msgs
from road_generation import schema

from base.vector import Vector  # Base class

__author__ = "Konstantin Ditschuneit"
__copyright__ = "KITcar"


class InvalidPointOperationError(Exception):
    pass


class Point(Vector):
    """Point subclass of Vector which implements a point.

    Compared with its Superclass, this class imposes some restrictions to better fit
    the interpretation of a point in the mathematical sense.

    Uses vector's initializer.

    """

    def to_geometry_msg(self):
        """To ROS geometry_msg.

        Returns:
            This point as a geometry_msgs/Point """
        return geometry_msgs.Point(x=self.x, y=self.y, z=self.z)

    def to_schema(self):
        """To schema Point.

        Mainly used in lanelets for road generation.

        Returns:
            This point as a schema Point."""
        return schema.point(x=self.x, y=self.y)

    def rotated(self, num):
        raise InvalidPointOperationError("A point cannot be rotated.")

    def __sub__(self, p):
        if not type(p) is Vector:
            raise InvalidPointOperationError("A point can only be modified by a vector.")
        return Point(self.x - p.x, self.y - p.y, self.z - p.z)

    def __add__(self, p):
        if not type(p) is Vector:
            raise InvalidPointOperationError("A point can only be modified by a vector.")
        return Point(self.x + p.x, self.y + p.y, self.z + p.z)

    def __rmul__(self, num):
        try:
            return Point(Vector(self).rotated(num.get_angle()) + Vector(num))
        except (NotImplementedError, AttributeError):
            pass

        raise InvalidPointOperationError("A point cannot be scaled.")
