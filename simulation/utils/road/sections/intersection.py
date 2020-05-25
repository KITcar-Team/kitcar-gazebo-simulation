"""Intersection."""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

from simulation.utils.geometry import Point, Line, Vector, Pose, Polygon

from simulation.utils.road.sections.road_section import RoadSection
from simulation.utils.road.sections.road_section import MarkedLine
from simulation.utils.road.config import Config
import simulation.utils.road.sections.type as road_section_type
from simulation.utils.road.sections.traffic_sign import TrafficSign
from simulation.utils.road.sections.surface_marking import SurfaceMarkingRect


def _get_stop_line(line1: Line, line2: Line, kind) -> SurfaceMarkingRect:
    """Return a line perpendicular to both provided (assumed parallel) lines.

    The returned line will be at the first point where both lines are parallel to each other plus 2cm offset.
    """
    beginning_line1 = line1.interpolate(0.02)
    beginning_line2 = line2.interpolate(0.02)

    # Test which line to draw the stop line at
    if beginning_line1.distance(line2) < beginning_line2.distance(line1):
        # End of line 1 is the starting point of the stop line
        p1 = beginning_line1
        p2 = line2.interpolate(line2.project(beginning_line1))
    else:
        # End of line 2 is the starting point
        p1 = line1.interpolate(line1.project(beginning_line2))
        p2 = beginning_line2

    line = Line([p1, p2])
    width = line.length
    center = 0.5 * (Vector(line.coords[0]) + Vector(line.coords[1]))
    angle = Pose([0, 0], Vector(line.coords[1]) - Vector(line.coords[0])).get_angle()

    return SurfaceMarkingRect(
        kind=kind, angle=angle, width=width, normalize_x=False, center=center, depth=0.04
    )


def arange_with_end(start: float, end: float, step: float):
    return np.arange(start, end + step, step)


@dataclass
class Intersection(RoadSection):
    """Road section representing an intersection.

    Args:
        angle (float) = math.pi/2: Angle [radian] between crossing roads.
        closing (int) = None: Optionally close one direction to create a T-intersection.
        turn (int) = Intersection.STRAIGHT: Direction in which the road continues.
        rule (int) = Intersection.EQUAL: Priority-rule at intersection.
        size (float) = 1.8: Length of the crossing roads.
    """

    TYPE = road_section_type.INTERSECTION

    STRAIGHT = 0
    """Possible value for :attr:`turn`. Drive straight through the intersection."""
    LEFT = 1
    """Possible value for :attr:`turn`. Turn left at the intersection."""
    RIGHT = 2
    """Possible value for :attr:`turn`. Turn right at the intersection."""

    EQUAL = 0
    """Possible value for :attr:`rule`. *Rechts vor links.*"""
    YIELD = 1
    """Possible value for :attr:`rule`. Car must yield."""
    STOP = 2
    """Possible value for :attr:`rule`. Car must stop."""
    PRIORITY_YIELD = 3
    """Possible value for :attr:`rule`. Car will have the right of way.

    Intersecting road must yield.
    """
    PRIORITY_STOP = 4
    """Possible value for :attr:`rule`. Car will have the right of way.

    Intersecting road must stop.
    """

    angle: float = math.pi / 2
    """Angle between intersecting roads [radian]."""
    closing: str = None
    """Closed direction (T-intersection)."""
    turn: str = STRAIGHT
    """Direction in which road continues."""
    rule: str = EQUAL
    """Priority rule at intersection."""
    size: float = 1.8
    """Size of intersection (from one side to the other)."""

    def __post_init__(self):
        self._alpha = self.angle - math.pi / 2
        self._closing = self.closing

        self._size = self.size / 2

        self.traffic_signs.extend(self._get_intersection_traffic_signs())
        self.surface_markings.extend(self._get_intersection_surface_markings())

        # Check if size is large enough
        assert (-1 * self.w + self.v).y > (-1 * self.u).y and self.z.x > (self.x - self.u).x

        super().__post_init__()

    # all vectors defined as a property are defined in the
    # local coordinate system!
    @property
    def sin(self):
        return math.sin(self._alpha)

    @property
    def cos(self):
        return math.cos(self._alpha)

    @property
    def y(self):
        return Vector(0, -Config.road_width)

    @property
    def x(self):
        return Vector(Config.road_width / self.cos, 0)

    @property
    def z(self):
        return Vector(self._size, 0)

    @property
    def u(self):
        return Vector(math.tan(self._alpha) * Config.road_width, -Config.road_width)

    @property
    def v(self):
        return Vector(
            Config.road_width * self.cos, Config.road_width * math.sin(self._alpha),
        )

    @property
    def w(self):
        return Vector(r=self._size, phi=-math.pi / 2 + self._alpha)

    @property
    def ll(self):
        return Vector(
            0,
            (
                -1 * self.y
                - self.x
                - self.u
                - (2 - 2 * self.sin) / (self.cos * self.cos) * self.v
            ).y,
        )

    @property
    def ls(self):
        return Vector(
            0, (-1 * self.u + (-1 + self.sin) / (self.cos * self.cos) * self.v).y,
        )

    @property
    def rl(self):
        return Vector(
            0,
            (
                self.x
                + self.u
                + self.y
                - (2 + 2 * self.sin) / (self.cos * self.cos) * self.v
            ).y,
        )

    @property
    def rs(self):
        return Vector(0, (self.u - (1 + self.sin) / (self.cos * self.cos) * self.v).y)

    # all center_points for signs and surface markings are defined in
    # the local coordinate system!
    # move cp_surface by Config.TURN_SF_MARK_WIDTH/2 because of rendering
    # TODO remove magic number sign length (0.1), determine real length
    def cp_sign_south(self, sign_dist):
        return Vector(self.z - self.x + self.u) - Vector(
            0.1 + sign_dist, Config.get_sign_road_padding()
        )

    def cp_surface_south(self):
        return Vector(self.z - self.x + 0.5 * self.u) - Vector(
            Config.get_surface_mark_dist(), Config.TURN_SF_MARK_WIDTH / 2
        )

    def cp_sign_west(self, sign_dist):
        return (
            Vector(self.z - self.x - self.u)
            - Vector((0.1 + sign_dist) * 1 / self.u.norm * self.u)
            - Vector(Config.get_sign_road_padding() * 1 / self.v.norm * self.v)
        )

    def cp_surface_west(self):
        return (
            Vector(self.z - 0.5 * self.x - self.u)
            - Vector(Config.get_surface_mark_dist() * 1 / self.u.norm * self.u)
            - Vector(Config.TURN_SF_MARK_WIDTH / 2 * 1 / self.v.norm * self.v)
        )

    def cp_sign_north(self, sign_dist):
        return Vector(self.z + self.x - self.u) + Vector(
            0.1 + sign_dist, Config.get_sign_road_padding()
        )

    def cp_sign_east(self, sign_dist):
        return (
            Vector(self.z + self.x + self.u)
            + Vector((0.1 + sign_dist) * 1 / self.u.norm * self.u)
            + Vector(Config.get_sign_road_padding() * 1 / self.v.norm * self.v)
        )

    def cp_surface_east(self):
        return (
            Vector(self.z + 0.5 * self.x + self.u)
            + Vector(Config.get_surface_mark_dist() * 1 / self.u.norm * self.u)
            + Vector(Config.TURN_SF_MARK_WIDTH / 2 * 1 / self.v.norm * self.v)
        )

    def get_points(self):
        return [
            self.get_beginning()[0],
            self.get_ending()[0],
        ]

    # south is origin
    @property
    def middle_line_south(self) -> Line:
        return self.transform * Line([Point(0, 0), Point(self.z - self.x)])

    @property
    def left_line_south(self) -> Line:
        return self.transform * Line(
            [Point(0, Config.road_width), Point(self.z - self.x - self.u)]
        )

    @property
    def right_line_south(self) -> Line:
        return self.transform * Line(
            [Point(0, -Config.road_width), Point(self.z - self.x + self.u)]
        )

    @property
    def middle_line_east(self) -> Line:
        return self.transform * Line([Point(self.z + self.u), Point(self.z + self.w)])

    @property
    def left_line_east(self) -> Line:
        return self.transform * Line(
            [Point(self.z + self.x + self.u), Point(self.z + self.w + self.v)]
        )

    @property
    def right_line_east(self) -> Line:
        return self.transform * Line(
            [Point(self.z - self.x + self.u), Point(self.z + self.w - self.v)]
        )

    @property
    def middle_line_north(self) -> Line:
        return self.transform * Line([Point(self.z + self.x), Point(2 * self.z)])

    @property
    def left_line_north(self) -> Line:
        return self.transform * Line(
            [Point(self.z + self.x - self.u), Point(2 * self.z - self.y)]
        )

    @property
    def right_line_north(self) -> Line:
        return self.transform * Line(
            [Point(self.z + self.x + self.u), Point(2 * self.z + self.y)]
        )

    @property
    def middle_line_west(self) -> Line:
        return self.transform * Line([Point(self.z - self.u), Point(self.z - self.w)])

    @property
    def left_line_west(self) -> Line:
        return self.transform * Line(
            [Point(self.z - self.x - self.u), Point(self.z - self.w - self.v)]
        )

    @property
    def right_line_west(self) -> Line:
        return self.transform * Line(
            [Point(self.z + self.x - self.u), Point(self.z - self.w + self.v)]
        )

    @property
    def ls_circle(self) -> Line:
        if self.turn == Intersection.LEFT:
            points_ls = []
            for theta in arange_with_end(0, 0.5 * math.pi + self._alpha, math.pi / 20):
                points_ls.append(Point(self.z - self.x + self.ls - self.ls.rotated(theta)))
            return self.transform * Line(points_ls)

    @property
    def ll_circle(self) -> Line:
        if self.turn == Intersection.LEFT:
            points_ll = []
            for theta in arange_with_end(0, 0.5 * math.pi + self._alpha, math.pi / 40):
                points_ll.append(
                    Point(self.z - self.x + self.u + self.ll - self.ll.rotated(theta))
                )
            return self.transform * Line(points_ll)

    @property
    def rs_circle(self) -> Line:
        if self.turn == Intersection.RIGHT:
            points_rs = []
            for theta in arange_with_end(0, -math.pi / 2 + self._alpha, -math.pi / 20):
                points_rs.append(Point(self.z - self.x + self.rs - self.rs.rotated(theta)))
            return self.transform * Line(points_rs)

    @property
    def rl_circle(self) -> Line:
        if self.turn == Intersection.RIGHT:
            points_rl = []
            for theta in arange_with_end(0, -math.pi / 2 + self._alpha, -math.pi / 40):
                points_rl.append(
                    Point(self.z - self.x - self.u + self.rl - self.rl.rotated(theta))
                )
            return self.transform * Line(points_rl)

    @property
    def middle_line(self) -> Line:
        """Line: Middle line of the intersection."""
        if self.turn == Intersection.LEFT:
            return self.middle_line_south + self.ls_circle + self.middle_line_west
        elif self.turn == Intersection.RIGHT:
            return self.middle_line_south + self.rs_circle + self.middle_line_east
        else:
            straight_m_l = Line(
                [
                    self.middle_line_south.get_points()[-1],
                    self.middle_line_north.get_points()[0],
                ]
            )
            return self.middle_line_south + straight_m_l + self.middle_line_north

    @property
    def lines(self) -> List[MarkedLine]:
        """List[MarkedLine]: All road lines with their marking type."""
        lines = []
        south_middle_end_length = self.prev_length + self.middle_line_south.length
        north_middle_start_length = -0.1
        north_left_start_length = -0.1
        north_right_start_length = -0.1
        west_middle_start_length = -0.1
        east_middle_start_length = -0.1

        if self.turn == Intersection.LEFT:
            lines.append(
                MarkedLine.from_line(
                    self.ls_circle, self.DASHED_LINE_MARKING, south_middle_end_length
                )
            )
            lines.append(
                MarkedLine.from_line(self.ll_circle, self.DASHED_LINE_MARKING, -0.1)
            )
            west_middle_start_length = south_middle_end_length + self.ls_circle.length
        elif self.turn == Intersection.RIGHT:
            lines.append(
                MarkedLine.from_line(
                    self.rs_circle, self.DASHED_LINE_MARKING, south_middle_end_length
                )
            )
            lines.append(
                MarkedLine.from_line(self.rl_circle, self.DASHED_LINE_MARKING, -0.1)
            )
            east_middle_start_length = south_middle_end_length + self.rs_circle.length
        else:
            north_middle_start_length = (
                south_middle_end_length
                + Line(
                    [
                        self.middle_line_south.get_points()[-1],
                        self.middle_line_north.get_points()[0],
                    ]
                ).length
            )
            north_left_start_length = (
                self.prev_length
                + self.left_line_south.length
                + Line(
                    [
                        self.left_line_south.get_points()[-1],
                        self.left_line_north.get_points()[0],
                    ]
                ).length
            )
            north_right_start_length = (
                self.prev_length
                + self.right_line_south.length
                + Line(
                    [
                        self.right_line_south.get_points()[-1],
                        self.right_line_north.get_points()[0],
                    ]
                ).length
            )

        # south + left west + right east
        lines.append(
            MarkedLine.from_line(
                self.left_line_south + self.left_line_west,
                self.left_line_marking,
                self.prev_length,
            )
        )
        lines.append(
            MarkedLine.from_line(
                self.middle_line_south, self.middle_line_marking, self.prev_length
            )
        )
        lines.append(
            MarkedLine.from_line(
                self.right_line_south + self.right_line_east,
                self.right_line_marking,
                self.prev_length,
            )
        )
        # west
        lines.append(
            MarkedLine.from_line(
                self.middle_line_west, self.middle_line_marking, west_middle_start_length
            )
        )
        lines.append(
            MarkedLine.from_line(
                self.right_line_west, self.right_line_marking, south_middle_end_length
            )
        )
        # north
        lines.append(
            MarkedLine.from_line(
                self.left_line_north, self.left_line_marking, north_left_start_length
            )
        )
        lines.append(
            MarkedLine.from_line(
                self.middle_line_north, self.middle_line_marking, north_middle_start_length
            )
        )
        lines.append(
            MarkedLine.from_line(
                self.right_line_north, self.right_line_marking, north_right_start_length
            )
        )
        # east
        lines.append(
            MarkedLine.from_line(
                self.left_line_east, self.left_line_marking, south_middle_end_length
            )
        )
        lines.append(
            MarkedLine.from_line(
                self.middle_line_east, self.middle_line_marking, east_middle_start_length
            )
        )

        return lines

    def get_beginning(self) -> Tuple[Pose, float]:
        return (Pose(self.transform * Point(0, 0), self.transform.get_angle() + math.pi), 0)

    def get_ending(self) -> Tuple[Pose, float]:
        if self.turn == Intersection.LEFT:
            end_angle = math.pi / 2 + self._alpha
            end_point = Point(self.z - self.w)
        elif self.turn == Intersection.RIGHT:
            end_angle = -math.pi / 2 + self._alpha
            end_point = Point(self.z + self.w)
        elif self.turn == Intersection.STRAIGHT:
            end_angle = 0
            end_point = Point(2 * self.z)

        return (Pose(self.transform * end_point, self.transform.get_angle() + end_angle), 0)

    def get_bounding_box(self) -> Polygon:
        """Get a polygon around the road section.

        Bounding box is an approximate representation of all points within a given distance \
        of this geometric object.
        """
        return Polygon(self.middle_line.buffer(1.5 * self.size))

    def _get_intersection_traffic_signs(self) -> List[TrafficSign]:
        signs = []
        if self.turn == Intersection.LEFT:
            # sign "turn left" in south
            signs.append(
                TrafficSign(
                    kind=TrafficSign.TURN_LEFT,
                    center=Point(self.cp_sign_south(Config.get_turn_sign_dist())),
                    angle=math.pi,
                )
            )
            if self.rule != Intersection.YIELD:
                # sign "turn right" in west
                signs.append(
                    TrafficSign(
                        kind=TrafficSign.TURN_RIGHT,
                        center=Point(self.cp_sign_west(Config.get_turn_sign_dist())),
                        angle=0.5 * math.pi + self._alpha,
                    )
                )
        elif self.turn == Intersection.RIGHT:
            # sign "turn right" in south
            signs.append(
                TrafficSign(
                    kind=TrafficSign.TURN_RIGHT,
                    center=Point(self.cp_sign_south(Config.get_turn_sign_dist())),
                    angle=math.pi,
                )
            )
            if self.rule != Intersection.YIELD:
                # sign "turn left" in east
                signs.append(
                    TrafficSign(
                        kind=TrafficSign.TURN_LEFT,
                        center=Point(self.cp_sign_east(Config.get_turn_sign_dist())),
                        angle=-0.5 * math.pi + self._alpha,
                    )
                )

        type_map = {
            Intersection.PRIORITY_YIELD: TrafficSign.PRIORITY,
            Intersection.PRIORITY_STOP: TrafficSign.PRIORITY,
            Intersection.YIELD: TrafficSign.YIELD,
            Intersection.STOP: TrafficSign.STOP,
        }
        if self.rule in type_map:
            signs.append(
                TrafficSign(
                    kind=type_map[self.rule],
                    center=Point(self.cp_sign_south(Config.get_prio_sign_dist(1))),
                    angle=math.pi,
                )
            )
            signs.append(
                TrafficSign(
                    kind=type_map[self.rule],
                    center=Point(self.cp_sign_north(Config.get_prio_sign_dist(1))),
                )
            )

        # stvo-206: Stoppschild,
        # stvo-306: Vorfahrtsstraße
        # todo: also add turning signal if we are not on the outer turn lane
        # on the opposite side
        type_map_opposite = {
            Intersection.PRIORITY_YIELD: TrafficSign.YIELD,
            Intersection.PRIORITY_STOP: TrafficSign.STOP,
            Intersection.YIELD: TrafficSign.PRIORITY,
            Intersection.STOP: TrafficSign.PRIORITY,
        }

        if self.rule in type_map_opposite:
            signs.append(
                TrafficSign(
                    kind=type_map_opposite[self.rule],
                    center=Point(self.cp_sign_west(Config.get_prio_sign_dist(1))),
                    angle=0.5 * math.pi + self._alpha,
                )
            )
            signs.append(
                TrafficSign(
                    kind=type_map_opposite[self.rule],
                    center=Point(self.cp_sign_east(Config.get_prio_sign_dist(1))),
                    angle=-0.5 * math.pi + self._alpha,
                )
            )

        for sign in signs:
            sign.transform = self.transform
            sign.normalize_x = False

        return signs

    def _get_intersection_surface_markings(self) -> List[SurfaceMarkingRect]:
        markings = []
        if self.turn == Intersection.LEFT or self.turn == Intersection.RIGHT:
            own_marking = (
                SurfaceMarkingRect.LEFT_TURN_MARKING
                if self.turn == Intersection.LEFT
                else SurfaceMarkingRect.RIGHT_TURN_MARKING
            )

            # roadmarking "turn left" in south
            markings.append(
                SurfaceMarkingRect(
                    kind=own_marking,
                    angle=0.5 * math.pi,
                    center=Point(self.cp_surface_south()),
                )
            )
            if self.rule != Intersection.YIELD:
                opposite_marking = (
                    SurfaceMarkingRect.RIGHT_TURN_MARKING
                    if self.turn == Intersection.LEFT
                    else SurfaceMarkingRect.LEFT_TURN_MARKING
                )
                opposite_angle = self._alpha + (
                    0 if self.turn == Intersection.LEFT else math.pi
                )
                opposite_center = Point(
                    self.cp_surface_west()
                    if self.turn == Intersection.LEFT
                    else self.cp_surface_east()
                )
                # roadmarking "turn right" in west
                markings.append(
                    SurfaceMarkingRect(
                        kind=opposite_marking, angle=opposite_angle, center=opposite_center,
                    )
                )

        # Add stop lines
        west_line = None
        east_line = None
        north_line = None
        south_line = None
        if self.rule == Intersection.EQUAL and self.turn == Intersection.STRAIGHT:
            west_line = SurfaceMarkingRect.GIVE_WAY_LINE
            east_line = SurfaceMarkingRect.GIVE_WAY_LINE
            north_line = SurfaceMarkingRect.GIVE_WAY_LINE
            south_line = SurfaceMarkingRect.GIVE_WAY_LINE
        elif (
            self.rule == Intersection.PRIORITY_YIELD and self.turn == Intersection.STRAIGHT
        ):
            west_line = SurfaceMarkingRect.GIVE_WAY_LINE
            east_line = SurfaceMarkingRect.GIVE_WAY_LINE
        elif self.rule == Intersection.PRIORITY_STOP and self.turn == Intersection.STRAIGHT:
            west_line = SurfaceMarkingRect.STOP_LINE
            east_line = SurfaceMarkingRect.STOP_LINE
        elif self.rule == Intersection.YIELD:
            north_line = SurfaceMarkingRect.GIVE_WAY_LINE
            south_line = SurfaceMarkingRect.GIVE_WAY_LINE
        elif self.rule == Intersection.STOP:
            north_line = SurfaceMarkingRect.STOP_LINE
            south_line = SurfaceMarkingRect.STOP_LINE

        # These stop lines are always the direction's middle and right line
        # going away from the center of the intersection in local coordinates
        if west_line is not None:
            markings.append(
                _get_stop_line(
                    Line([Point(self.z - self.u), Point(self.z - self.w)]),
                    Line(
                        [Point(self.z - self.x - self.u), Point(self.z - self.w - self.v)]
                    ),
                    kind=west_line,
                )
            )
        if north_line is not None:
            markings.append(
                _get_stop_line(
                    Line(
                        [Point(self.z + self.x), Point(2 * self.z)]
                    ),  # Middle line north in local coords
                    Line(
                        [Point(self.z + self.x - self.u), Point(2 * self.z - self.y)]
                    ),  # Right line
                    kind=north_line,
                )
            )
        if south_line is not None:
            markings.append(
                _get_stop_line(
                    Line([Point(self.z - self.x), Point(0, 0)]),
                    Line([Point(self.z - self.x + self.u), Point(0, -Config.road_width)]),
                    kind=south_line,
                )
            )
        if east_line is not None:
            markings.append(
                _get_stop_line(
                    Line([Point(self.z + self.u), Point(self.z + self.w)]),
                    Line(
                        [Point(self.z + self.x + self.u), Point(self.z + self.w + self.v)]
                    ),
                    kind=east_line,
                )
            )

        for marking in markings:
            marking.transform = self.transform
            marking.normalize_x = False

        return markings
