"""Road elements are simple individual components of the road that have a frame.

Examples are traffic signs, obstacles or surface markings (e.g. turn arrow on the ground.
"""

from dataclasses import dataclass
from typing import Union

from simulation.utils.geometry import Line, Point, Polygon, Transform
from simulation.utils.road.sections.transformable import Transformable


@dataclass
class RoadElement(Transformable):
    normalize_x: bool = True
    """If true, all x-values are substracted by the lowest x-value."""

    _frame: Polygon = Polygon(
        [Point(-0.1, -0.2), Point(0.1, -0.2), Point(0.1, 0.2), Point(-0.1, 0.2)]
    )
    """Polygon: Frame of the element in local coordinates."""

    def set_transform(self, obj: Union[Line, Transform]):
        """Calculate the correct transform to this element.

        Depending on :attr:`self.normalize_x` the positional behavior is different. If
        :attr:`self.normalize_x` is True, the element is aligned along the provided line.
        """
        if type(obj) is Line:
            pose = obj.interpolate_pose(
                arc_length=self._center.x if self.normalize_x else 0
            )
            obj = Transform(pose, pose.get_angle())
        super().set_transform(obj)

    @property
    def frame(self) -> Polygon:
        """Polygon: Frame of the element in global coordinates."""
        tf = (
            Transform([-self._center.x, 0], 0) if self.normalize_x else Transform([0, 0], 0)
        )
        return self.transform * (tf * self._frame)

    @property
    def center(self) -> Point:
        """Point: Center point of the element in global coordinates."""
        return Point(self.frame.centroid)

    @property
    def _center(self) -> Point:
        """Point: Center point of the element in local coordinates."""
        return Point(self._frame.centroid)


@dataclass
class RoadElementRect(RoadElement):
    """Generic element of the road that has a frame.

    Examples of road elements are obstacles and traffic signs.

    Args:
        arc_length: x coordinate of the element along the road.
        y: y coordinate of the element. (Perpendicular to the road.)
        width: Width of the element.
        depth: Depth of the element. Component of the size in the direction of the road.
        angle: Angle [radian] between the middle line and the element
            (measured at the center).
    """

    width: float = 0.2
    """Width of the element."""
    depth: float = 0.2
    """Depth (length) of the element."""
    angle: float = 0
    """Angle [radian] between the middle line and the element (measured at the center)."""

    def __init__(
        self,
        arc_length: float = 0.4,
        y: float = -0.2,
        width: float = width,
        depth: float = depth,
        angle: float = angle,
        normalize_x: bool = True,
        z: float = 0,
        height: float = 0,
    ):
        """Initialize a retangular road element."""
        for obj in arc_length, y, width, depth, angle:
            assert isinstance(obj, float) or isinstance(
                obj, int
            ), f"Should be a number but is {obj}"

        self.width = width
        self.depth = depth
        self.angle = angle
        self.height = height
        super().__init__(
            normalize_x=normalize_x,
            _frame=Transform(Point(arc_length, y, z), self.angle)
            * Polygon(
                [
                    [-self.depth / 2, self.width / 2],
                    [self.depth / 2, self.width / 2],
                    [self.depth / 2, -self.width / 2],
                    [-self.depth / 2, -self.width / 2],
                ]
            ),
        )

    @property
    def orientation(self) -> float:
        """float: Orientation of the element in global coordinates in radians."""
        return self.transform.get_angle() + self.angle
