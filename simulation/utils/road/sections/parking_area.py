"""ParkingArea, ParkingLot, ParkingSpot and StartLine."""

import itertools
import math
from dataclasses import dataclass, field
from typing import List

import simulation.utils.road.sections.type as road_section_type
from simulation.utils.geometry import Line, Point, Polygon, Transform
from simulation.utils.road.config import Config
from simulation.utils.road.sections import (
    ParkingObstacle,
    RoadSection,
    StraightRoad,
    SurfaceMarkingRect,
    TrafficSign,
)
from simulation.utils.road.sections.road_element import RoadElementRect
from simulation.utils.road.sections.road_section import MarkedLine
from simulation.utils.road.sections.transformable import Transformable


@dataclass
class ParkingSpot(RoadElementRect):
    """Parking spot with a type and optionally an obstacle placed on top.

    Args:
        width: Width of the spot.
        kind: Type of the spot.
        obstacle: Obstacle within the spot.
    """

    FREE = 0
    """Possible value of :attr:`kind`."""
    OCCUPIED = 1
    """Possible value of :attr:`kind`."""
    BLOCKED = 2
    """Possible value of :attr:`kind`."""

    _side: str = None
    """Side of the road."""
    kind: str = FREE
    """Classification of the parking spot."""
    obstacle: ParkingObstacle = None
    """Obstacle within the spot."""
    x_surface_marking: SurfaceMarkingRect = None

    def __init__(
        self,
        kind: float = kind,
        width: float = 0.35,
        obstacle: ParkingObstacle = obstacle,
    ):
        self.kind = kind
        self.obstacle = obstacle
        super().__init__(width=width, normalize_x=False)

        if self.kind == ParkingSpot.BLOCKED:
            self.x_surface_marking = SurfaceMarkingRect(
                SurfaceMarkingRect.PARKING_SPOT_X,
                *self._center.xy,
                width=self.width,
                depth=self.depth,
                angle=0,
                normalize_x=False,
            )

    def set_transform(self, tf: Transform):
        self._frame = Polygon(
            [
                [0, 0],
                [self.depth, 0],
                [self.depth, -self.width],
                [0, -self.width],
            ]
        )
        super().set_transform(tf)

        if self.obstacle is not None:
            self.obstacle.set_transform(self.transform)

        if self.x_surface_marking is not None:
            self.x_surface_marking.width = self.width
            self.x_surface_marking.depth = self.depth
            self.x_surface_marking._frame = self._frame
            self.x_surface_marking.set_transform(self.transform)

    @property
    def lines(self) -> List[MarkedLine]:
        """List[MarkedLine]: Borderlines for spot if spot is on the left.

        Marking type is always solid.
        """

        lines = []
        if self._side == ParkingLot.LEFT_SIDE or self.kind == self.BLOCKED:
            spot_points = self.frame.get_points()
            left_border = Line(spot_points[:2])
            right_border = Line(spot_points[2:4])
            lines.append(MarkedLine.from_line(left_border, RoadSection.SOLID_LINE_MARKING))
            lines.append(MarkedLine.from_line(right_border, RoadSection.SOLID_LINE_MARKING))
        return lines


@dataclass
class ParkingLot(Transformable):
    """Outline of a parking lot (right/left side) and all parking spots contained within.

    The origin is invariant of the side of the road (left, right).
    It is always in the left corner of the left most parking spot,
    with the x-direction pointing away from the road.

    Args:
        start (float): Beginning relative to the start of the section.
        opening_angle (float): Opening angle of the outside border of the parking lot.
        depth (float): Depth of the parking spots within the parking lot.
        spots (List[ParkingSpot]): Parking spots within the lot.
    """

    RIGHT_SIDE = "right"
    """Possible value of :attr:`side`. Parking lot is on the left side of the road."""
    LEFT_SIDE = "left"
    """Possible value of :attr:`side`. Parking lot is on the right side of the road."""

    DEFAULT_LEFT_DEPTH = 0.5
    """Default value for the depth of parking spots on the left side."""
    DEFAULT_RIGHT_DEPTH = 0.3
    """Default value for the depth of parking spots on the right side."""

    start: float = 0
    """Start of the parking lot along the middle line relative to the parking area."""
    spots: List[ParkingSpot] = field(default_factory=list)
    """Parking spots within this parking lot."""
    _side: str = RIGHT_SIDE
    """Side of the road."""
    opening_angle: float = math.radians(60)
    """Opening angle of parking lot."""
    depth: float = None
    """Depth of parking spots within this lot.

    If no other value is provided, the default depth of parking lots is 0.5m on the left
    side and 0.3m on the right side.
    """

    def set_transform(self, new_tf: Transform):
        super().set_transform(new_tf)

        if self.depth is None:
            return

        spot_x = 0
        """X-Value of left lower border point."""
        spot_y = 0  # -self.depth / math.tan(self.opening_angle)
        """Y-Value of left lower border point."""

        for spot in (
            reversed(self.spots) if self._side == ParkingLot.RIGHT_SIDE else self.spots
        ):
            # Calculate local spot coordinate system
            spot.depth = self.depth
            spot._side = self._side
            spot.set_transform(self.transform * Transform([spot_x, spot_y], 0))

            spot_y -= spot.width

    @property
    def length(self) -> float:
        """float: Sum of the widths of all parking spots."""
        return sum(spot.width for spot in self.spots)

    @property
    def border(self) -> Line:
        """Line: Outside border of the parking lot."""
        if self.depth is None:
            return Line()

        border_points = [
            Point(0, self.depth / math.tan(self.opening_angle)),
            Point(
                self.depth,
                0,
            ),
            Point(
                self.depth,
                -self.length,
            ),
            Point(
                0,
                -self.depth / math.tan(self.opening_angle) - self.length,
            ),
        ]
        border = Line(
            reversed(border_points)
            if self._side == ParkingLot.RIGHT_SIDE
            else border_points
        )
        return self.transform * border

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
class ParkingArea(StraightRoad):
    """Part of the road with parking lots and a start line.

    Args:
        left_lots (List[ParkingLot]): Parking lots on the left side.
        right_lots (List[ParkingLot]): Parking lots on the right side.
        start_line (bool): Indicate whether the parking area starts with a start line.
        start_line_length (float): Manually set the length of the start line.
    """

    TYPE = road_section_type.PARKING_AREA

    start_line: bool = False
    """If the parking area has a start line."""
    start_line_length: float = 0.06
    """Length of the start line (if one is added."""
    left_lots: List[ParkingLot] = field(default_factory=list)
    """Parking lots on the left side."""
    right_lots: List[ParkingLot] = field(default_factory=list)
    """Parking lots on the right side."""

    def __post_init__(self):
        for lot in self.left_lots:
            lot._side = ParkingLot.LEFT_SIDE
            if lot.depth is None:
                lot.depth = ParkingLot.DEFAULT_LEFT_DEPTH

        for lot in self.right_lots:
            lot._side = ParkingLot.RIGHT_SIDE
            if lot.depth is None:
                lot.depth = ParkingLot.DEFAULT_RIGHT_DEPTH

        super().__post_init__()

        if self.start_line:
            # Create a start line.
            self.surface_markings.append(
                SurfaceMarkingRect(
                    kind=SurfaceMarkingRect.START_LINE,
                    arc_length=self.start_line_length / 2,
                    y=0,
                    normalize_x=False,
                    depth=self.start_line_length,
                    width=2 * Config.road_width,
                    angle=0,
                )
            )

        for lot in itertools.chain(self.left_lots, self.right_lots):
            for spot in lot.spots:
                if spot.x_surface_marking is not None:
                    self.surface_markings.append(spot.x_surface_marking)

        for obs in self.parking_obstacles:
            self.obstacles.append(obs)

        if len(self.right_lots) + len(self.left_lots) > 0:
            self.traffic_signs.append(
                TrafficSign(
                    kind=TrafficSign.PARKING,
                    arc_length=0,
                    y=-Config.road_width - 0.1,
                    angle=0,
                    normalize_x=False,
                )
            )

    def set_transform(self, new_tf: Transform):
        super().set_transform(new_tf)
        for lot in self.left_lots:
            # Set transform to first spot
            lot.set_transform(
                self.transform
                * Transform(
                    [
                        lot.start + lot.depth / math.tan(lot.opening_angle),
                        Config.road_width,
                    ],
                    math.pi / 2,
                )
            )
        for lot in self.right_lots:
            # Set transform to last spot
            lot.set_transform(
                self.transform
                * Transform(
                    [
                        lot.start + lot.length + lot.depth / math.tan(lot.opening_angle),
                        -Config.road_width,
                    ],
                    -math.pi / 2,
                )
            )

    def get_bounding_box(self) -> Polygon:
        """Get a polygon around the road section.

        Bounding box is an approximate representation of all points within a given distance
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
        for lot in itertools.chain(self.left_lots, self.right_lots):
            lines.extend(lot.lines)
        return lines
