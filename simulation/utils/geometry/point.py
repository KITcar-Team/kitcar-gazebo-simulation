"""Basic point class which is compatible with all needed formats."""

__copyright__ = "KITcar"

# Compatible formats
import geometry_msgs.msg as geometry_msgs

from simulation.utils.geometry.transform import Transform
from simulation.utils.geometry.vector import Vector  # Base class

from .frame import validate_and_maintain_frames


class InvalidPointOperationError(Exception):
    pass


class Point(Vector):
    """Point subclass of Vector which implements a point.

    Compared with its Superclass, this class imposes some restrictions to better fit
    the interpretation of a point in the mathematical sense.

    Uses vector's initializer.
    """

    def to_geometry_msg(self) -> geometry_msgs.Point:
        """To ROS geometry_msg.

        Returns:
            This point as a geometry_msgs/Point
        """
        return geometry_msgs.Point32(x=self.x, y=self.y, z=self.z)

    def rotated(self, *args, **kwargs):
        raise InvalidPointOperationError("A point cannot be rotated.")

    @validate_and_maintain_frames
    def __sub__(self, p):
        if not type(p) is Vector:
            raise InvalidPointOperationError("A point can only be modified by a vector.")
        return super().__sub__(p)

    @validate_and_maintain_frames
    def __add__(self, p):
        if not type(p) is Vector:
            raise InvalidPointOperationError("A point can only be modified by a vector.")
        return super().__add__(p)

    @validate_and_maintain_frames
    def __rmul__(self, obj: Transform):
        """Right multiplication of a point.

        Only defined for transformations.
        """
        if type(obj) == Transform:
            # Transform * self
            return self.__class__(obj * Vector(self))

        raise InvalidPointOperationError("A point cannot be scaled.")
