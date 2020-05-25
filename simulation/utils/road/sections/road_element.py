"""Road elements are simple individual components of the road that have a frame.

Examples are traffic signs, obstacles or surface markings (e.g. turn arrow on the ground.
"""

from dataclasses import dataclass

from simulation.utils.geometry import Polygon, Point, Transform, Vector, Line


@dataclass
class RoadElement:
    transform: Transform = None
    """Transform to coordinate system in which frame is given."""
    normalize_x: bool = True
    """If true, all x-values are substracted by the lowest x-value."""

    def __post_init__(self):
        # prevents execution when building documentation
        if self.transform is None:
            self.transform = Transform([0, 0], 0)

    def set_transform(self, line: Line):
        """Calculate the correct transform to this element.

        Depending on :attr:`self.normalize_x` the positional behavior is different.
        If :attr:`self.normalize_x` is True, the element is aligned along the provided line.

        Example:
            >>> from simulation.utils.geometry import Line, Point, Transform
            >>> from simulation.utils.road.sections.road_element import RoadElementRect
            >>> line = Line([Point(0, 0), Point(0, 10)])  # y axis
            >>> normalized_el = RoadElementRect(center=Point(1, 1))  # normalize_x is True by default
            >>> normalized_el.set_transform(line)
            >>> normalized_el.transform
            Transform(translation=(0.0, 1.0, 0.0),rotation=90.0 degrees)
            >>> normalized_el._center
            Point(1.0, 1.0, 0.0)
            >>> normalized_el.center
            Point(-1.0, 1.0, 0.0)
            >>> unnormalized_el = RoadElementRect(center=Point(1,0), normalize_x=False)
            ... # normalize_x is True by default
            >>> unnormalized_el.set_transform(line)
            >>> unnormalized_el.transform
            Transform(translation=(0.0, 0.0, 0.0),rotation=90.0 degrees)
            >>> normalized_el._center
            Point(1.0, 1.0, 0.0)
            >>> normalized_el.center
            Point(-1.0, 1.0, 0.0)
        """
        pose = line.interpolate_pose(arc_length=self._center.x if self.normalize_x else 0)
        self.transform = Transform(pose, pose.get_angle())


@dataclass
class _RoadElementPoly(RoadElement):
    frame: Polygon = Polygon(
        [Point(0.3, -0.4), Point(0.5, -0.4), Point(0.5, 0), Point(0.3, 0)]
    )
    """Polygon: Frame of the element in global coordinates."""

    def __post_init__(self):
        super().__post_init__()
        self._center = self.frame.centroid


@dataclass
class RoadElementPoly(_RoadElementPoly):
    @property
    def frame(self) -> Polygon:
        """Polygon: Frame of the element in global coordinates."""
        return self.transform * self._frame

    @frame.setter
    def frame(self, poly: Polygon):
        self._frame = poly


@dataclass
class _RoadElementRect(RoadElement):
    center: Point = Point(0.4, -0.2)
    """Center point of the element."""
    width: float = 0.2
    """Width of the element."""
    depth: float = 0.2
    """Depth of the element.

    Component of the size in the direction of the road.
    """
    angle: float = 0
    """Angle [radian] between the middle line and the element (measured at the center)."""


@dataclass
class RoadElementRect(_RoadElementRect):
    """Generic element of the road that has a frame.

    Examples of road elements are obstacles and traffic signs.
    """

    @property
    def center(self) -> Point:
        """Point: Center of the element in global coordinates."""
        tf = Transform([-self._center.x, 0], 0) if self.normalize_x else 1
        return Point(self.transform * (tf * Vector(self._center)))

    @center.setter
    def center(self, c: Point):
        if not type(c) is Point:
            c = Point(c)
        self._center = c

    @property
    def orientation(self) -> float:
        """float: Orientation of the element in global coordinates in radians."""
        return self.transform.get_angle() + self.angle

    @property
    def frame(self) -> Polygon:
        """Polygon: Frame of the element in global coordinates."""
        return Transform(self.center, self.orientation) * Polygon(
            [
                [-self.depth / 2, self.width / 2],
                [self.depth / 2, self.width / 2],
                [self.depth / 2, -self.width / 2],
                [-self.depth / 2, -self.width / 2],
            ]
        )
