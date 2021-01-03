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
    _center: Point = Point(0, 0)
    """Center of the object in local coordinates."""

    _frame: Polygon = Polygon(
        [Point(0.3, -0.4), Point(0.5, -0.4), Point(0.5, 0), Point(0.3, 0)]
    )
    """Polygon: Frame of the element in global coordinates."""

    def set_transform(self, obj: Union[Line, Transform]):
        """Calculate the correct transform to this element.

        Depending on :attr:`self.normalize_x` the positional behavior is different.
        If :attr:`self.normalize_x` is True, the element is aligned along the provided line.

        Example:
            >>> from simulation.utils.geometry import Line, Point, Transform
            >>> from simulation.utils.road.sections.road_element import RoadElementRect
            >>> line = Line([Point(0, 0), Point(0, 10)])  # y axis
            >>> normalized_el = RoadElementRect(_center=Point(1, 1))
            ... # normalize_x is True by default
            >>> normalized_el.set_transform(line)
            >>> normalized_el.transform
            Transform(translation=Vector(0.0, 1.0, 0.0),\
rotation=Quaternion(0.7071067811865476, 0.0, 0.0, 0.7071067811865475))
            >>> normalized_el._center
            Point(1.0, 1.0, 0.0)
            >>> normalized_el.center
            Point(-1.0, 1.0, 0.0)
            >>> unnormalized_el = RoadElementRect(_center=Point(1,0), normalize_x=False)
            ... # normalize_x is True by default
            >>> unnormalized_el.set_transform(line)
            >>> unnormalized_el.transform
            Transform(translation=Vector(0.0, 0.0, 0.0),\
rotation=Quaternion(0.7071067811865476, 0.0, 0.0, 0.7071067811865475))
            >>> normalized_el._center
            Point(1.0, 1.0, 0.0)
            >>> normalized_el.center
            Point(-1.0, 1.0, 0.0)
        """
        if type(obj) is Line:
            pose = obj.interpolate_pose(
                arc_length=self._center.x if self.normalize_x else 0
            )
            obj = Transform(pose, pose.get_angle())
        super().set_transform(obj)

    @property
    def frame(self) -> Polygon:
        tf = (
            Transform([-self._center.x, 0], 0) if self.normalize_x else Transform([0, 0], 0)
        )
        return self.transform * (tf * self._frame)

    @property
    def center(self) -> Point:
        return Point(self.frame.centroid)


@dataclass
class RoadElementRect(RoadElement):
    """Generic element of the road that has a frame.

    Examples of road elements are obstacles and traffic signs.
    """

    _center: Point = Point(0.4, -0.2)
    """Center point of the element in local coordinates."""
    width: float = 0.2
    """Width of the element."""
    depth: float = 0.2
    """Depth of the element.

    Component of the size in the direction of the road.
    """
    angle: float = 0
    """Angle [radian] between the middle line and the element (measured at the center)."""

    def __post_init__(self):
        self._frame = Transform(self._center, self.angle) * Polygon(
            [
                [-self.depth / 2, self.width / 2],
                [self.depth / 2, self.width / 2],
                [self.depth / 2, -self.width / 2],
                [-self.depth / 2, -self.width / 2],
            ]
        )
        super().__post_init__()

    @property
    def orientation(self) -> float:
        """float: Orientation of the element in global coordinates in radians."""
        return self.transform.get_angle() + self.angle
