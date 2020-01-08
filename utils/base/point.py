"""
Basic point class which is compatible with all needed formats

Author: Konstantin Ditschuneit
Date: 15.11.2019
"""

#### Compatible formats ####
import geometry_msgs.msg as geometry_msgs
from road_generation import schema

from base.vector import Vector  # Base class


class InvalidPointOperationError(Exception):
    pass


class Point(Vector):

    def __init__(self, *args, **kwargs):
        """ Initialize point from @args 
            
            @args can be one of the following types:
                - np.array
                - geometry_msgs/Point(32)
                - tuple of x,y(,z) coordinates
            @kwargs can contain
                - polar coordinates: length 'r' and angle 'phi'
        """

        try:
            super(Point, self).__init__(*args, **kwargs)
        except:
            #None of the initializations worked
            raise NotImplementedError(
                f"Point initialization not implemented for {type(args[0])}")

    def to_geometry_msg(self):
        """ Convert point to geometry_msgs/Point """
        return geometry_msgs.Point(x=self.x, y=self.y, z=self.z)

    def to_schema(self):
        """ Convert point to schema.point"""
        return schema.point(x=self.x, y=self.y)

    def rotated(self, num):
        raise InvalidPointOperationError("A point cannot be rotated.")

    def __sub__(self, p):
        if not type(p) is Vector:
            raise InvalidPointOperationError(
                "A point can only be modified by a vector.")
        return Point(self.x - p.x, self.y - p.y, self.z - p.z)

    def __add__(self, p):
        if not type(p) is Vector:
            raise InvalidPointOperationError(
                "A point can only be modified by a vector.")
        return Point(self.x + p.x, self.y + p.y, self.z + p.z)

    def __rmul__(self, num):
        try:
            return Point(Vector(self).rotated(num.get_angle()) + Vector(num))
        except:
            pass

        raise InvalidPointOperationError("A point cannot be scaled.")
