"""ParkingArea, ParkingLot, ParkingSpot and StartLine."""

import math
from dataclasses import dataclass, field
from typing import List
import itertools

from simulation.utils.geometry import Point, Vector, Line, Polygon, Transform

from simulation.utils.road.config import Config
from simulation.utils.road.sections.road_section import MarkedLine
import simulation.utils.road.sections.type as road_section_type
from simulation.utils.road.sections import StraightRoad
from simulation.utils.road.sections import ParkingObstacle
from simulation.utils.road.sections import SurfaceMarkingRect
from simulation.utils.road.sections import RoadSection


@dataclass
class _ParkingSpot:

    FREE = 0
    """Possible value of :attr:`kind`."""
    OCCUPIED = 1
    """Possible value of :attr:`kind`."""
    BLOCKED = 2
    """Possible value of :attr:`kind`."""

    width: float = 0.4
    """Width of parking spot."""
    _depth: float = field(default=0, init=False)
    """Depth of parking spot."""
    _side: str = field(default=None, init=False)
    """Side of the road."""
    kind: str = FREE
    """Classification of the parking spot."""
    obstacle: ParkingObstacle = None
    """Obstacle within the spot."""
    transform: Transform = None
    """Transform to origin of parking spot.

    The origin is invariant of the side of the road (left, right).
    It is always in the left corner close to the road,
    with the x-direction pointing away from the road.
    """

    def __post_init__(self):
        # prevents execution when building documentation
        if self.transform is None:
            self.transform = Transform([0, 0], 0)


class ParkingSpot(_ParkingSpot):
    """Parking spot with a type and optionally an obstacle placed on top.

    Args:
        width (float): Width of the spot.
        kind (int) = ParkingSpot.FREE: Type of the spot.
        obstacle (ParkingObstacle) = None: Obstacle within the spot.
    """

    @property
    def frame(self) -> Polygon:
        """Polygon: Frame of the parking spot in global coordinates."""
        poly = Polygon(
            [
                Point(0, 0),
                Point(self._depth, 0),
                Point(self._depth, -self.width),
                Point(0, -self.width),
            ]
        )
        return self.transform * poly

    @property
    def lines(self) -> List[MarkedLine]:
        """List[MarkedLine]: Borderlines for spot if spot is on the left.

        Marking type is always solid."""
        lines = []
        if self._side == ParkingLot.LEFT_SIDE or self.kind == self.BLOCKED:
            spot_points = self.frame.get_points()
            left_border = Line(spot_points[:2])
            right_border = Line(spot_points[2:4])
            lines.append(MarkedLine.from_line(left_border, RoadSection.SOLID_LINE_MARKING))
            lines.append(MarkedLine.from_line(right_border, RoadSection.SOLID_LINE_MARKING))
        return lines

    @property
    def obstacle(self) -> ParkingObstacle:
        """ParkingObstacle: Obstacle that is on the spot (if any)."""
        if self._obstacle is not None:
            self._obstacle.transform = self.transform
        return self._obstacle

    @obstacle.setter
    def obstacle(self, obs):
        self._obstacle = obs


@dataclass
class _ParkingLot:

    RIGHT_SIDE = "right"
    """Possible value of :attr:`side`. Parking lot is on the left side of the road."""
    LEFT_SIDE = "left"
    """Possible value of :attr:`side`. Parking lot is on the right side of the road."""

    start: float = 0
    """Start of the parking lot along the middle line relative to the parking area."""
    spots: List[ParkingSpot] = field(default_factory=list)
    """Parking spots within this parking lot."""
    _side: str = RIGHT_SIDE
    """Side of the road."""
    opening_angle: float = math.radians(60)
    """Opening angle of parking lot."""
    depth: float = 0.4
    """Depth of parking spots within this lot."""
    transform: Transform = None
    """Transform to origin of the parking lot.

    The origin is invariant of the side of the road (left, right).
    It is always in the left most corner of the left most parking spot
    close to the road, with the x-direction pointing away from the road.
    """

    def __post_init__(self):
        # prevents execution when building documentation
        if self.transform is None:
            self.transform = Transform([0, 0], 0)


class ParkingLot(_ParkingLot):
    """Outline of a parking lot (right/left side) and all parking spots contained within.

    Args:
        start (float): Beginning relative to the start of the section.
        opening_angle (float): Opening angle of the outside border of the parking lot.
        depth (float): Depth of the parking spots within the parking lot.
        spots (List[ParkingSpot]): Parking spots within the lot.

    """

    @property
    def length(self) -> float:
        """float: Sum of the widths of all parking spots."""
        return sum(spot.width for spot in self._spots)

    @property
    def _side_sign(self) -> float:
        # If the parking lot is on the left side, Y coordinates are mirrored
        # i.e. multiplied by -1
        return 1 if self._side == "left" else -1

    @property
    def border(self) -> Line:
        """Line: Outside border of the parking lot."""
        border = Line(
            [
                Point(self.start, self._side_sign * Config.road_width),
                Point(
                    self.start + self.depth / math.tan(self.opening_angle),
                    self._side_sign * (self.depth + Config.road_width),
                ),
                Point(
                    self.start + self.depth / math.tan(self.opening_angle) + self.length,
                    self._side_sign * (self.depth + Config.road_width),
                ),
                Point(
                    self.start
                    + self.length
                    + 2 * self.depth / math.tan(self.opening_angle),
                    self._side_sign * Config.road_width,
                ),
            ]
        )
        return self.transform * border

    @property
    def spots(self) -> List[ParkingSpot]:
        """List[ParkingSpot]: All spots in the parking lot."""
        spot_x = self.start + self.depth / math.tan(self.opening_angle)
        """X-Value of left upper border point."""
        spot_y = self._side_sign * Config.road_width
        """Y-Value of left lower border point."""

        for spot in self._spots:
            angle = self.transform.get_angle()
            # Calculate local spot coordinate system
            # Local coordinate system is left on the right side of the road
            if self._side == "right":
                # Add math.pi if on right side
                angle += math.pi * 3 / 2
                origin = Vector(spot_x + spot.width, spot_y)
            else:
                angle += math.pi / 2
                origin = Vector(spot_x, spot_y)

            spot.transform = Transform(self.transform * origin, angle)
            spot._depth = self.depth
            spot._side = self._side

            spot_x += spot.width

        return self._spots

    @spots.setter
    def spots(self, spts):
        self._spots = spts

    @property
    def obstacles(self) -> List[ParkingObstacle]:
        """List[ParkingObstacle]: All obstacles on spots."""
        return [spot.obstacle for spot in self.spots if spot.obstacle is not None]

    @property
    def lines(self) -> List[MarkedLine]:
        """List[MarkedLine]: All border lines with solid marking type."""
        lines = []
        lines.append(MarkedLine.from_line(self.border, RoadSection.SOLID_LINE_MARKING))
        for spot in self.spots:
            lines.extend(spot.lines)
        return lines


@dataclass
class _ParkingArea(StraightRoad):

    TYPE = road_section_type.PARKING_AREA

    start_line: bool = False
    """If the parking area has a start line."""
    start_line_length: float = 0.06
    """Length of the start line (if one is added."""
    left_lots: List[ParkingLot] = field(default_factory=list)
    """Parking lots on the left side."""
    right_lots: List[ParkingLot] = field(default_factory=list)
    """Parking lots on the right side."""


class ParkingArea(_ParkingArea):
    """Part of the road with parking lots and a start line.

    Args:
        left_lots (List[ParkingLot]): Parking lots on the left side.
        right_lots (List[ParkingLot]): Parking lots on the right side.
        start_line (bool): Indicate whether the parking area starts with a start line.
        start_line_length (float): Manually set the length of the start line.
    """

    @property
    def surface_markings(self) -> List[SurfaceMarkingRect]:
        markings = []
        if self.start_line:
            # Create a start line.
            markings.append(
                SurfaceMarkingRect(
                    center=self.transform * Point(self.start_line_length / 2, 0),
                    normalize_x=False,
                    depth=self.start_line_length,
                    width=2 * Config.road_width,
                    kind=SurfaceMarkingRect.START_LINE,
                    angle=self.transform.get_angle(),
                )
            )

        for lot in itertools.chain(self.left_lots, self.right_lots):
            for spot in lot.spots:
                if spot.kind == ParkingSpot.BLOCKED:
                    c = spot.frame.centroid
                    markings.append(
                        SurfaceMarkingRect(
                            width=spot.width,
                            depth=spot._depth,
                            kind=SurfaceMarkingRect.PARKING_SPOT_X,
                            center=Point(c.x, c.y),
                            angle=spot.transform.get_angle(),
                            normalize_x=False,
                        )
                    )
        return super().surface_markings + markings

    @surface_markings.setter
    def surface_markings(self, markings: List[SurfaceMarkingRect]):
        self._surface_markings = markings

    def get_bounding_box(self) -> Polygon:
        """Get a polygon around the road section.

        Bounding box is an approximate representation of all points within a given distance \
        of this geometric object.
        """
        biggest_depth = 0
        for ll, rl in zip(self.left_lots, self.right_lots):
            if ll.depth > biggest_depth:
                biggest_depth = ll.depth
            if rl.depth > biggest_depth:
                biggest_depth = rl.depth
        return Polygon(self.middle_line.buffer(1.5 * (biggest_depth + Config.road_width)))

    @property
    def left_lots(self) -> List[ParkingLot]:
        """List[ParkingLot]: Parking lots on the left side."""
        for lot in self._left_lots:
            lot.transform = self.transform
            lot._side = ParkingLot.LEFT_SIDE
        return self._left_lots

    @left_lots.setter
    def left_lots(self, ll: List[ParkingLot]):
        self._left_lots = ll

    @property
    def right_lots(self) -> List[ParkingLot]:
        """List[ParkingLot]: Parking lots on the right side."""
        for lot in self._right_lots:
            lot.transform = self.transform
            lot._side = ParkingLot.RIGHT_SIDE
        return self._right_lots

    @right_lots.setter
    def right_lots(self, rl: List[ParkingLot]):
        self._right_lots = rl

    @property
    def parking_obstacles(self) -> List[ParkingObstacle]:
        """List[ParkingObstacle]: All obstacles on parking spots."""
        return sum(
            (lot.obstacles for lot in itertools.chain(self.left_lots, self.right_lots)), []
        )

    @property
    def lines(self) -> List[MarkedLine]:
        """List[MarkedLine]: All borderlines with their marking type."""
        lines = []
        lines.extend(super().lines)
        for lot in self.left_lots:
            lines.extend(lot.lines)
        for lot in self.right_lots:
            lines.extend(lot.lines)
        return lines
