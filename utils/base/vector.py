"""
Basic vector class which is compatible with all needed formats

Author: Konstantin Ditschuneit
Date: 15.11.2019
"""

import shapely.geometry  # Base class
import math  # For calculations

#### Compatible formats ####
import numpy as np
import geometry_msgs.msg as geometry_msgs
from road_generation import schema

import numbers

class Vector(shapely.geometry.point.Point):

    def __init__(self, *args, **kwargs):
        """ Initialize vector from @args 
            
            @args can be one of the following types:
                - np.array
                - geometry_msgs/Vector(32)
                - tuple of x,y(,z) coordinates
            @kwargs can contain
                - polar coordinates: length 'r' and angle 'phi'

            A vector is always initialized with 3 coordinates. If there's no third coordinate provided, z:=0.
        """

        #Try to add z component
        try:
            if len(args) == 2:
                args = (*args, 0)
        except:
            pass
        try:
            if len(args[0]) == 2:
                args = (*args[0], 0)
        except:
            pass

        try:
            # construct Vector from r, phi
            r = kwargs['r']
            phi = kwargs['phi']
            self.__init__(math.cos(phi)*r, math.sin(phi)*r)
            return
        except:
            pass

        #Attempt default init
        try:
            super(Vector, self).__init__(*args)
            return
        except:
            pass

        #Try to initialize geometry vector
        try:
            # Call this function with values extracted
            self.__init__(args[0].x, args[0].y, args[0].z)
            return
        except:
            pass
        #None of the initializations worked
        raise NotImplementedError(
            f"Vector initialization not implemented for {type(args[0])}")

    def to_geometry_msg(self):
        """ Convert to geometry_msgs/Vector """
        return geometry_msgs.Vector3(x=self.x, y=self.y, z=self.z)

    def to_numpy(self):
        """ Return as a numpy array. """
        return np.array([self.x, self.y, self.z])

    @property
    def norm(self):
        """ Calculate the arithmetic norm of the vector. """
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def rotated(self, angle):
        """ Returns the vector rotated by @angle given in radian around [0,0,0] in the groundplane. """
        c = math.cos(angle)
        s = math.sin(angle)
        return Vector(c*self.x - s*self.y, s*self.x + c *
                         self.y, self.z)  # Matrix multiplication

    def __sub__(self, p):
        p1 = Vector(self.x - p.x, self.y - p.y, self.z - p.z)
        return p1

    def __add__(self, p):
        p1 = Vector(self.x + p.x, self.y + p.y, self.z + p.z)
        return p1

    def __mul__(self, vec):
        if isinstance(vec, Vector):
            return vec.x * self.x + vec.y * self.y + vec.z * self.z

        return NotImplemented

    def __rmul__(self, num):
        try:
            return self.rotated(num.get_angle()) + Vector(num)
        except:
            pass

        if isinstance(num, numbers.Number):
            return Vector(num*self.x, num*self.y, num*self.z)

        return NotImplemented

    def __eq__(self, vector):
        TOLERANCE = 1e-8
        return type(vector) == type(self) \
            and (Vector(vector)-Vector(self)).norm < TOLERANCE
