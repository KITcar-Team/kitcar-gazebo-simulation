"""ParkingArea, ParkingLot, ParkingSpot and StartLine."""

import math
from dataclasses import dataclass, field
from typing import List
import itertools

from geometry import Point, Vector, Line, Polygon, Transform

from road.config import Config
from road.sections.road_section import Export
import road.sections.type as road_section_type
from road.sections import StraightRoad
from road.sections import ParkingObstacle
from road import schema


@dataclass
class StartLine:
    """Object representing a start line."""

    length: float = 0.06
    """Length in direction of the road."""
    transform: Transform = None
    """Transform to the start line.

    The transform points to the beginning in direction of the road
    and in the middle of the startline perpendicular to the road.
    """

    def __post_init__(self):
        # prevents execution when building documentation
        if self.transform is None:
            self.transform = Transform([0, 0], 0)

    @property
    def frame(self) -> Polygon:
        """Polygon: Frame of the start line."""
        # polygon.to_lanelet reverses the points in left boundary
        poly = Polygon(
            [
                Point(0, -Config.road_width),
                Point(0, Config.road_width),
                Point(self.length, Config.road_width),
                Point(self.length, -Config.road_width),
            ]
        )
        return self.transform * poly

    def export(self) -> Export:
        startLane = self.frame.to_schema_lanelet()
        startLane.type = "startLane"
        return startLane


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
    def obstacle(self) -> ParkingObstacle:
        """ParkingObstacle: Obstacle that is on the spot (if any)."""
        if self._obstacle is not None:
            self._obstacle.transform = self.transform
        return self._obstacle

    @obstacle.setter
    def obstacle(self, obs):
        self._obstacle = obs

    def export(self) -> Export:
        lanelet = self.frame.to_schema_lanelet()
        if self._side == "left" or self.kind == ParkingSpot.BLOCKED:
            lanelet.leftBoundary.lineMarking = "solid"
            lanelet.rightBoundary.lineMarking = "solid"
        if self.kind == ParkingSpot.BLOCKED:
            lanelet.type = "parking_spot_x"

        if self.obstacle:
            return [lanelet, self.obstacle.export()]
        else:
            return [lanelet]


@dataclass
class _ParkingLot:
    """Outline of a parking lot (right/left side) and all parking spots contained within."""

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
    opening_angle: float = 60
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

        self.opening_angle = math.radians(self.opening_angle)


class ParkingLot(_ParkingLot):
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

    def export(self) -> Export:
        # Export outline of parking lot
        border = self.border.get_points()
        inner_boundary = Line([border[0], border[-1]]).to_schema_boundary()
        outer_boundary = self.border.to_schema_boundary()
        outer_boundary.lineMarking = "solid"

        left_boundary = inner_boundary if self._side == "right" else outer_boundary
        right_boundary = outer_boundary if self._side == "right" else inner_boundary

        lanelet_list = []
        lanelet_list.extend(
            [schema.lanelet(leftBoundary=left_boundary, rightBoundary=right_boundary)]
        )

        for spot in self.spots:
            lanelet = spot.export()
            lanelet_list.extend(lanelet)

        return lanelet_list


@dataclass
class _ParkingArea(StraightRoad):

    TYPE = road_section_type.PARKING_AREA

    start_line: StartLine = None
    left_lots: List[ParkingLot] = field(default_factory=list)
    """Parking lots on the left side."""
    right_lots: List[ParkingLot] = field(default_factory=list)
    """Parking lots on the right side."""


class ParkingArea(_ParkingArea):
    """Part of the road with parking lots and a start line."""

    @property
    def start_line(self) -> StartLine:
        """StartLine: When provided, start line of the parking area."""
        if self._start_line is not None:
            self._start_line.transform = self.transform
        return self._start_line

    @start_line.setter
    def start_line(self, sl: StartLine):
        self._start_line = sl

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

    def export(self) -> Export:
        # Defines a straight line
        export = super().export()

        # Include start line
        if self._start_line is not None:
            export.objects.append(self.start_line.export())

        # Add left and right lot
        for ll, rl in zip(self.left_lots, self.right_lots):
            export.objects.extend(ll.export())
            export.objects.extend(rl.export())
        return export
