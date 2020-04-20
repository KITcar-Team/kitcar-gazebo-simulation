"""The RoadSection is parent to all other RoadSection classes."""

from dataclasses import dataclass, field
from typing import List, Tuple
import math

from geometry import Transform, Polygon, Line, Pose

from road.config import Config
from road.sections import StaticObstacle
from road import schema


class Export:
    def __init__(self, lanelet1, lanelet2, others=[]):
        self.objects = [lanelet1, lanelet2]
        self.objects.extend(others)
        self.lanelet_pairs = [(lanelet1, lanelet2)]


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
    def obstacles(self) -> List[StaticObstacle]:
        """List[StaticObstacle]: All obstacles within this section of the road."""
        for obstacle in self._obstacles:
            pose = self.middle_line.interpolate_pose(arc_length=obstacle._center.x)
            obstacle.transform = Transform(pose, pose.get_angle())
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obs: List[StaticObstacle]):
        self._obstacles = obs

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

    def export(self) -> Export:
        lanelet1 = schema.lanelet(
            leftBoundary=schema.boundary(), rightBoundary=schema.boundary()
        )
        lanelet2 = schema.lanelet(
            leftBoundary=schema.boundary(), rightBoundary=schema.boundary()
        )
        lanelet1.isStart = self.is_start
        lanelet1.leftBoundary = self.middle_line.to_schema_boundary()
        lanelet1.rightBoundary = self.right_line.to_schema_boundary()
        lanelet2.leftBoundary = self.middle_line.to_schema_boundary()
        lanelet2.rightBoundary = self.left_line.to_schema_boundary()
        # reverse boundary of left lanelet to match driving direction
        lanelet2.leftBoundary.point.reverse()
        lanelet2.rightBoundary.point.reverse()

        lanelet1.rightBoundary.lineMarking = self.right_line_marking
        lanelet1.leftBoundary.lineMarking = self.middle_line_marking
        lanelet2.rightBoundary.lineMarking = self.left_line_marking

        return Export(lanelet1, lanelet2, others=[obs.export() for obs in self.obstacles])
