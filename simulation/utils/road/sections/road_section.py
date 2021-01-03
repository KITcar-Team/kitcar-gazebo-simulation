"""The RoadSection is parent to all other RoadSection classes."""
import functools
import itertools
import math
from dataclasses import dataclass, field
from typing import List, Tuple

from simulation.utils.geometry import Line, Point, Polygon, Pose, Transform
from simulation.utils.road.config import Config
from simulation.utils.road.sections import StaticObstacle, SurfaceMarking, TrafficSign
from simulation.utils.road.sections.speed_limit import SpeedLimit
from simulation.utils.road.sections.transformable import Transformable


class MarkedLine(Line):
    """Line with a defined line marking style."""

    def __init__(self, *args, **kwargs):
        assert "style" in kwargs
        if "prev_length" not in kwargs:
            kwargs["prev_length"] = 0
        self.style = kwargs["style"]
        self.prev_length = kwargs["prev_length"]
        del kwargs["style"]
        del kwargs["prev_length"]

        super().__init__(*args, **kwargs)

    @classmethod
    def from_line(cls, line: Line, style, prev_length=0):
        m = cls(style=style, prev_length=prev_length)
        m._set_coords(line.coords)
        return m

    def __repr__(self) -> str:
        return super().__repr__()[:-1] + f", style={self.style})"


@dataclass
class RoadSection(Transformable):
    """Base class of all road sections."""

    SOLID_LINE_MARKING = "solid"
    """Continuous white line."""
    DASHED_LINE_MARKING = "dashed"
    """Dashed white line."""
    MISSING_LINE_MARKING = "missing"
    """No line at all."""

    id: int = 0
    """Road section id (consecutive integers by default)."""
    is_start: bool = False
    """Road section is beginning of the road."""

    left_line_marking: str = SOLID_LINE_MARKING
    """Marking type of the left line."""
    middle_line_marking: str = DASHED_LINE_MARKING
    """Marking type of the middle line."""
    right_line_marking: str = SOLID_LINE_MARKING
    """Marking type of the right line."""
    obstacles: List[StaticObstacle] = field(default_factory=list)
    """Obstacles in the road section."""
    traffic_signs: List[TrafficSign] = field(default_factory=list)
    """Traffic signs in the road section."""
    surface_markings: List[SurfaceMarking] = field(default_factory=list)
    """Surface markings in the road section."""
    _speed_limits: List[SpeedLimit] = field(default_factory=list)
    """Speed limits in the road section."""
    TYPE = None
    """Type of the road section."""
    prev_length: float = 0
    """Length of Road up to this section."""

    def __post_init__(self):
        assert (
            self.__class__.TYPE is not None
        ), "Subclass of RoadSection missing TYPE declaration!"

        super().__post_init__()
        self.set_transform(self.transform)

    def set_transform(self, tf):
        # Invalidate cached middle line
        if self.__dict__.get("middle_line"):
            self.__dict__.pop("middle_line")
        super().set_transform(tf)
        for obj in itertools.chain(
            self.obstacles, self.surface_markings, self.traffic_signs
        ):
            if obj.normalize_x:
                obj.set_transform(self.middle_line)
            else:
                obj.set_transform(self.transform)

    @functools.cached_property
    def middle_line(self) -> Line:
        """Line: Middle line of the road section."""
        return Line()

    @property
    def left_line(self) -> Line:
        """Line: Left line of the road section."""
        return self.middle_line.parallel_offset(Config.road_width, "left")

    @property
    def right_line(self) -> Line:
        """Line: Right line of the road section."""
        return self.middle_line.parallel_offset(Config.road_width, "right")

    @property
    def lines(self) -> List[MarkedLine]:
        """List[MarkedLine]: All road lines with their marking type."""
        lines = []
        lines.append(
            MarkedLine.from_line(self.left_line, self.left_line_marking, self.prev_length)
        )
        lines.append(
            MarkedLine.from_line(
                self.middle_line, self.middle_line_marking, self.prev_length
            )
        )
        lines.append(
            MarkedLine.from_line(self.right_line, self.right_line_marking, self.prev_length)
        )
        return lines

    @property
    def speed_limits(self) -> List[SpeedLimit]:
        """Speed limits in the road section."""
        return self._speed_limits

    def get_bounding_box(self) -> Polygon:
        """Get a polygon around the road section.

        Bounding box is an approximate representation of all points within a given distance
        of this geometric object.
        """
        return Polygon(self.middle_line.buffer(1.5 * Config.road_width))

    def get_beginning(self) -> Tuple[Pose, float]:
        """Get the beginning of the section as a pose and the curvature.

        Returns:
            A tuple consisting of the first point on the middle line together with \
            the direction facing away from the road section as a pose and the curvature \
            at the beginning of the middle line.
        """
        pose = Transform([0, 0], math.pi) * self.middle_line.interpolate_pose(arc_length=0)
        curvature = self.middle_line.interpolate_curvature(arc_length=0)

        return (pose, curvature)

    def get_ending(self) -> Tuple[Pose, float]:
        """Get the ending of the section as a pose and the curvature.

        Returns:
            A tuple consisting of the last point on the middle line together with \
            the direction facing along the middle line as a pose and the curvature \
            at the ending of the middle line.
        """
        pose = self.middle_line.interpolate_pose(arc_length=self.middle_line.length)
        curvature = self.middle_line.interpolate_curvature(
            arc_length=self.middle_line.length
        )

        return (pose, curvature)

    def add_speed_limit(self, arc_length: float, speed: int):
        """Add a speed limit to this road section.

        Args:
            arc_length: Direction along the road to the speed limit.
            speed: Speed limit. Negative values correspond to the end of a speed limit zone.
        """
        speed_limit = SpeedLimit(arc_length, limit=speed)
        sm = speed_limit.surface_marking
        ts = speed_limit.traffic_sign
        sm.set_transform(self.middle_line)
        ts.set_transform(self.middle_line)
        self.speed_limits.append(speed_limit)
        self.surface_markings.append(sm)
        self.traffic_signs.append(ts)

        return ts

    def add_obstacle(
        self,
        arc_length: float = 0.2,
        y_offset: float = -0.2,
        angle: float = 0,
        width: float = 0.2,
        length: float = 0.3,
        height: float = 0.25,
    ):
        """Add an obstacle to the road.

        Args:
        arc_length: Direction along the road to the obstacle.
        y_offset: Offset orthogonal to the middle line.
        angle: Orientation offset of the obstacle.
        width: Width of the obstacle.
        length: Length of the obstacle.
        height: Heigth of the obstacle.
        """
        o = StaticObstacle(
            _center=Point(arc_length, y_offset),
            angle=angle,
            width=width,
            depth=length,
            height=height,
        )
        o.set_transform(self.middle_line)
        self.obstacles.append(o)

        return o
