"""The RoadSection is parent to all other RoadSection classes."""

from dataclasses import dataclass, field
from typing import List, Tuple
import math

from simulation.utils.geometry import Transform, Polygon, Line, Pose

from simulation.utils.road.config import Config
from simulation.utils.road.sections import StaticObstacle, TrafficSign, SurfaceMarking


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
class _RoadSection:
    SOLID_LINE_MARKING = "solid"
    """Continuous white line."""
    DASHED_LINE_MARKING = "dashed"
    """Dashed white line."""
    MISSING_LINE_MARKING = "missing"
    """No line at all."""

    id: int = 0
    """Road section id (consecutive integers by default)."""
    transform: Transform = None
    """Transform to origin of the road section."""
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

    def __post_init__(self):
        assert (
            self.__class__.TYPE is not None
        ), "Subclass of RoadSection missing TYPE declaration!"

        if self.transform is None:
            self.transform = Transform([0, 0], 0)


class RoadSection(_RoadSection):
    """Base class of all road sections."""

    TYPE = None
    """Type of the road section."""
    prev_length: float = 0
    """Length of Road up to this section."""

    @property
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
    def obstacles(self) -> List[StaticObstacle]:
        """List[StaticObstacle]: All obstacles within this section of the road."""
        for obstacle in self._obstacles:
            obstacle.set_transform(self.middle_line)
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obs: List[StaticObstacle]):
        self._obstacles = obs

    @property
    def traffic_signs(self) -> List[TrafficSign]:
        """List[TrafficSign]: All traffic signs within this section of the road."""
        for sign in self._traffic_signs:
            sign.set_transform(self.middle_line)
        return self._traffic_signs

    @traffic_signs.setter
    def traffic_signs(self, signs: List[TrafficSign]):
        self._traffic_signs = signs

    @property
    def surface_markings(self) -> List[SurfaceMarking]:
        """List[SurfaceMarking]: All surface markings within this section of the road."""
        for marking in self._surface_markings:
            marking.set_transform(self.middle_line)
        return self._surface_markings

    @surface_markings.setter
    def surface_markings(self, markings: List[SurfaceMarking]):
        self._surface_markings = markings

    def get_bounding_box(self) -> Polygon:
        """Get a polygon around the road section.

        Bounding box is an approximate representation of all points within a given distance \
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
