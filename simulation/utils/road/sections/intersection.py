"""Intersection."""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from geometry import Point, Polygon, Line, Vector, Transform, Pose

from road.sections.road_section import RoadSection
from road.sections.road_section import Export
from road.config import Config
import road.sections.type as road_section_type
from road import schema


@dataclass
class Intersection(RoadSection):
    """Road section representing an intersection."""

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

    angle: float = 90
    """Angle between intersecting roads."""
    closing: str = None
    """Closed direction (T-intersection)."""
    turn: str = STRAIGHT
    """Direction in which road continues."""
    rule: str = EQUAL
    """Priority rule at intersection."""
    size: float = 1.8
    """Size of intersection (from one side to the other)."""

    def __post_init__(self):
        self._alpha = math.radians(self.angle - 90)
        self._closing = self.closing

        self._size = self.size / 2

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
            for theta in np.arange(0, 0.5 * math.pi + self._alpha, math.pi / 20):
                points_ls.append(Point(self.z - self.x + self.ls - self.ls.rotated(theta)))
            return self.transform * Line(points_ls)

    @property
    def ll_circle(self) -> Line:
        if self.turn == Intersection.LEFT:
            points_ll = []
            for theta in np.arange(0, 0.5 * math.pi + self._alpha, math.pi / 40):
                points_ll.append(
                    Point(self.z - self.x + self.u + self.ll - self.ll.rotated(theta))
                )
            return self.transform * Line(points_ll)

    @property
    def rs_circle(self) -> Line:
        if self.turn == Intersection.RIGHT:
            points_rs = []
            for theta in np.arange(0, -math.pi / 2 + self._alpha, -math.pi / 20):
                points_rs.append(Point(self.z - self.x + self.rs - self.rs.rotated(theta)))
            return self.transform * Line(points_rs)

    @property
    def rl_circle(self) -> Line:
        if self.turn == Intersection.RIGHT:
            points_rl = []
            for theta in np.arange(0, -math.pi / 2 + self._alpha, -math.pi / 40):
                points_rl.append(
                    Point(self.z - self.x - self.u + self.rl - self.rl.rotated(theta))
                )
            return self.transform * Line(points_rl)

    @property
    def middle_line(self) -> Line:
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
    def arrow_left_south(self) -> Polygon:
        poly = Polygon(
            [
                Point(
                    self.cp_surface_south()
                    + Vector(-Config.TURN_SF_MARK_LENGTH, Config.TURN_SF_MARK_WIDTH)
                ),
                Point(self.cp_surface_south() + Vector(-Config.TURN_SF_MARK_LENGTH, 0)),
                Point(self.cp_surface_south() + Vector(0, 0)),
                Point(self.cp_surface_south() + Vector(0, Config.TURN_SF_MARK_WIDTH)),
            ]
        )
        return self.transform * poly

    @property
    def arrow_right_west(self) -> Polygon:
        poly = Polygon(
            [
                Point(Config.TURN_SF_MARK_WIDTH, Config.TURN_SF_MARK_LENGTH),
                Point(0, Config.TURN_SF_MARK_LENGTH),
                Point(0, 0),
                Point(Config.TURN_SF_MARK_WIDTH, 0),
            ]
        )
        t = Transform(
            self.transform * self.cp_surface_west(),
            self.transform.get_angle() + self._alpha,
        )
        return t * poly

    @property
    def arrow_right_south(self) -> Polygon:
        poly = Polygon(
            [
                Point(
                    self.cp_surface_south()
                    + Vector(-Config.TURN_SF_MARK_LENGTH, Config.TURN_SF_MARK_WIDTH,)
                ),
                Point(self.cp_surface_south() + Vector(-Config.TURN_SF_MARK_LENGTH, 0)),
                Point(self.cp_surface_south() + Vector(0, 0)),
                Point(self.cp_surface_south() + Vector(0, Config.TURN_SF_MARK_WIDTH)),
            ]
        )
        return self.transform * poly

    @property
    def arrow_left_east(self) -> Polygon:
        poly = Polygon(
            [
                Point(Config.TURN_SF_MARK_WIDTH, Config.TURN_SF_MARK_LENGTH),
                Point(0, Config.TURN_SF_MARK_LENGTH),
                Point(0, 0),
                Point(Config.TURN_SF_MARK_WIDTH, 0),
            ]
        )
        t = Transform(
            self.transform * self.cp_surface_east(),
            self.transform.get_angle() + self._alpha + math.pi,
        )
        return t * poly

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

    def export_direction(
        self, left, middle, right
    ) -> Tuple[schema.lanelet, schema.lanelet]:
        right_lanelet = schema.lanelet(
            leftBoundary=middle.to_schema_boundary(),
            rightBoundary=right.to_schema_boundary(),
        )
        right_lanelet.rightBoundary.lineMarking = "solid"
        right_lanelet.leftBoundary.lineMarking = "dashed"
        left_lanelet = schema.lanelet(
            leftBoundary=middle.to_schema_boundary(),
            rightBoundary=left.to_schema_boundary(),
        )
        left_lanelet.leftBoundary.point.reverse()
        left_lanelet.rightBoundary.point.reverse()
        left_lanelet.rightBoundary.lineMarking = "solid"
        return (right_lanelet, left_lanelet)

    def export(self):
        # Check if size is large enough
        assert (-1 * self.w + self.v).y > (-1 * self.u).y and self.z.x > (self.x - self.u).x

        others = []  # [southRight, southLeft, ...]
        last_pair = ()

        southRight, southLeft = self.export_direction(
            self.left_line_south, self.middle_line_south, self.right_line_south
        )

        eastRight, eastLeft = self.export_direction(
            self.left_line_east, self.middle_line_east, self.right_line_east
        )

        northRight, northLeft = self.export_direction(
            self.left_line_north, self.middle_line_north, self.right_line_north
        )

        westRight, westLeft = self.export_direction(
            self.left_line_west, self.middle_line_west, self.right_line_west
        )

        if self.rule == Intersection.EQUAL and self.turn == Intersection.STRAIGHT:
            westLeft.stopLine = "dashed"
            southRight.stopLine = "dashed"
            northLeft.stopLine = "dashed"
            eastLeft.stopLine = "dashed"
        elif (
            self.rule == Intersection.PRIORITY_YIELD and self.turn == Intersection.STRAIGHT
        ):
            westLeft.stopLine = "dashed"
            eastLeft.stopLine = "dashed"
        elif self.rule == Intersection.PRIORITY_STOP and self.turn == Intersection.STRAIGHT:
            westLeft.stopLine = "solid"
            eastLeft.stopLine = "solid"
        elif self.rule == Intersection.YIELD:
            southRight.stopLine = "dashed"
            northLeft.stopLine = "dashed"
        elif self.rule == Intersection.STOP:
            southRight.stopLine = "solid"
            northLeft.stopLine = "solid"

        closing_lanelet = schema.lanelet(
            leftBoundary=schema.boundary(), rightBoundary=schema.boundary()
        )
        closing_lanelet.rightBoundary.lineMarking = "solid"
        if self.closing == Intersection.STRAIGHT:
            closing_lanelet.rightBoundary.point.append(
                self.right_line_north.get_points()[0].to_schema()
            )
            closing_lanelet.rightBoundary.point.append(
                self.left_line_north.get_points()[0].to_schema()
            )
            closing_lanelet.leftBoundary.point.append(
                self.right_line_north.get_points()[0].to_schema()
            )
            closing_lanelet.leftBoundary.point.append(
                self.left_line_north.get_points()[0].to_schema()
            )
            others.append(closing_lanelet)
        else:
            others.append(northRight)
            others.append(northLeft)

        if self.closing == Intersection.LEFT:
            closing_lanelet.rightBoundary.point.append(
                self.right_line_west.get_points()[0].to_schema()
            )
            closing_lanelet.rightBoundary.point.append(
                self.left_line_west.get_points()[0].to_schema()
            )
            closing_lanelet.leftBoundary.point.append(
                self.right_line_west.get_points()[0].to_schema()
            )
            closing_lanelet.leftBoundary.point.append(
                self.left_line_west.get_points()[0].to_schema()
            )
            others.append(closing_lanelet)
        else:
            others.append(westRight)
            others.append(westLeft)

        if self.closing == Intersection.RIGHT:
            closing_lanelet.rightBoundary.point.append(
                self.right_line_east.get_points()[0].to_schema()
            )
            closing_lanelet.rightBoundary.point.append(
                self.left_line_east.get_points()[0].to_schema()
            )
            closing_lanelet.leftBoundary.point.append(
                self.right_line_east.get_points()[0].to_schema()
            )
            closing_lanelet.leftBoundary.point.append(
                self.left_line_east.get_points()[0].to_schema()
            )
            others.append(closing_lanelet)
        else:
            others.append(eastRight)
            others.append(eastLeft)

        if self.turn == Intersection.LEFT:
            turn_lines_right = schema.lanelet(
                leftBoundary=self.ls_circle.to_schema_boundary(),
                rightBoundary=self.ll_circle.to_schema_boundary(),
            )
            turn_lines_right.rightBoundary.lineMarking = "dashed"
            turn_lines_right.leftBoundary.lineMarking = "dashed"

            turn_lines_left = schema.lanelet(
                leftBoundary=self.ls_circle.to_schema_boundary(),
                rightBoundary=schema.boundary(),
            )
            turn_lines_left.leftBoundary.point.reverse()
            turn_lines_left.rightBoundary.point.append(
                (self.transform * Point(self.z - self.x - self.u)).to_schema()
            )

            others.append(turn_lines_right)
            others.append(turn_lines_left)

            # sign "turn left" in south
            sign = schema.trafficSign(
                type="stvo-209-10",
                orientation=self.transform.get_angle() + math.pi,
                centerPoint=(
                    self.transform * Point(self.cp_sign_south(Config.get_turn_sign_dist()))
                ).to_schema(),
            )
            others.append(sign)
            # roadmarking "turn left" in south
            road_marking = schema.roadMarking(
                type=schema.roadMarkingType.turn_right,
                orientation=self.transform.get_angle() + 0.5 * math.pi,
                centerPoint=(self.transform * Point(self.cp_surface_south())).to_schema(),
            )
            others.append(road_marking)

            if self.rule != Intersection.YIELD:
                # sign "turn right" in west
                sign = schema.trafficSign(
                    type="stvo-209-20",
                    orientation=self.transform.get_angle() + 0.5 * math.pi + self._alpha,
                    centerPoint=(
                        self.transform
                        * Point(self.cp_sign_west(Config.get_turn_sign_dist()))
                    ).to_schema(),
                )
                others.append(sign)
                # roadmarking "turn right" in west
                road_marking = schema.roadMarking(
                    type=schema.roadMarkingType.turn_left,
                    orientation=self.transform.get_angle() + self._alpha,
                    centerPoint=(
                        self.transform * Point(self.cp_surface_west())
                    ).to_schema(),
                )
                others.append(road_marking)

            last_pair = (westRight, westLeft)

        elif self.turn == Intersection.RIGHT:
            turn_lines_right = schema.lanelet(
                leftBoundary=schema.boundary(), rightBoundary=schema.boundary()
            )
            turn_lines_right.leftBoundary.lineMarking = "dashed"
            turn_lines_right = schema.lanelet(
                leftBoundary=self.rs_circle.to_schema_boundary(),
                rightBoundary=schema.boundary(),
            )
            turn_lines_right.rightBoundary.point.append(
                (self.transform * Point(self.z - self.x + self.u)).to_schema()
            )

            turn_lines_left = schema.lanelet(
                leftBoundary=self.rs_circle.to_schema_boundary(),
                rightBoundary=self.rl_circle.to_schema_boundary(),
            )
            turn_lines_left.leftBoundary.point.reverse()
            turn_lines_left.leftBoundary.point.reverse()
            turn_lines_left.rightBoundary.lineMarking = "dashed"
            turn_lines_left.leftBoundary.lineMarking = "dashed"

            others.append(turn_lines_right)
            others.append(turn_lines_left)

            # sign "turn right" in south
            sign = schema.trafficSign(
                type="stvo-209-20",
                orientation=self.transform.get_angle() + math.pi,
                centerPoint=(
                    self.transform * Point(self.cp_sign_south(Config.get_turn_sign_dist()))
                ).to_schema(),
            )
            others.append(sign)
            # roadmarking "turn right" in south
            road_marking = schema.roadMarking(
                type=schema.roadMarkingType.turn_left,
                orientation=self.transform.get_angle() + 0.5 * math.pi,
                centerPoint=(self.transform * Point(self.cp_surface_south())).to_schema(),
            )
            others.append(road_marking)

            if self.rule != Intersection.YIELD:
                # sign "turn left" in east
                sign = schema.trafficSign(
                    type="stvo-209-10",
                    orientation=self.transform.get_angle() - 0.5 * math.pi + self._alpha,
                    centerPoint=(
                        self.transform
                        * Point(self.cp_sign_east(Config.get_turn_sign_dist()))
                    ).to_schema(),
                )
                others.append(sign)
                # roadmarking "turn left" in east
                road_marking = schema.roadMarking(
                    type=schema.roadMarkingType.turn_right,
                    orientation=self.transform.get_angle() - math.pi + self._alpha,
                    centerPoint=(
                        self.transform * Point(self.cp_surface_east())
                    ).to_schema(),
                )
                others.append(road_marking)

            last_pair = (eastRight, eastLeft)

        elif self.turn == Intersection.STRAIGHT:
            straight_m_l = Line(
                [
                    self.middle_line_south.get_points()[-1],
                    self.middle_line_north.get_points()[0],
                ]
            )
            straight_l_l = Line(
                [
                    self.left_line_south.get_points()[-1],
                    self.left_line_north.get_points()[0],
                ]
            )
            straight_r_l = Line(
                [
                    self.right_line_south.get_points()[-1],
                    self.right_line_north.get_points()[0],
                ]
            )

            turn_lines_right = schema.lanelet(
                leftBoundary=straight_m_l.to_schema_boundary(),
                rightBoundary=straight_r_l.to_schema_boundary(),
            )
            turn_lines_left = schema.lanelet(
                leftBoundary=straight_m_l.to_schema_boundary(),
                rightBoundary=straight_l_l.to_schema_boundary(),
            )
            turn_lines_left.rightBoundary.point.reverse()
            turn_lines_left.leftBoundary.point.reverse()
            others.append(turn_lines_right)
            others.append(turn_lines_left)

            last_pair = (northRight, northLeft)

        type_map = {
            Intersection.PRIORITY_YIELD: "stvo-306",
            Intersection.PRIORITY_STOP: "stvo-306",
            Intersection.YIELD: "stvo-205",
            Intersection.STOP: "stvo-206",
        }
        if self.rule in type_map:
            others.append(
                schema.trafficSign(
                    type=type_map[self.rule],
                    orientation=self.transform.get_angle() + math.pi,
                    centerPoint=(
                        self.transform
                        * Point(self.cp_sign_south(Config.get_prio_sign_dist(1)))
                    ).to_schema(),
                )
            )
            others.append(
                schema.trafficSign(
                    type=type_map[self.rule],
                    orientation=self.transform.get_angle(),
                    centerPoint=(
                        self.transform
                        * Point(self.cp_sign_north(Config.get_prio_sign_dist(1)))
                    ).to_schema(),
                )
            )

        # stvo-206: Stoppschild,
        # stvo-306: Vorfahrtsstraße
        # todo: also add turning signal if we are not on the outer turn lane
        # on the opposite side
        type_map_opposite = {
            Intersection.PRIORITY_YIELD: "stvo-206",
            Intersection.PRIORITY_STOP: "stvo-306",
            Intersection.YIELD: "stvo-306",
            Intersection.STOP: "stvo-306",
        }

        if self.rule in type_map_opposite:
            others.append(
                schema.trafficSign(
                    type=type_map_opposite[self.rule],
                    orientation=self.transform.get_angle() + 0.5 * math.pi + self._alpha,
                    centerPoint=(
                        self.transform
                        * Point(self.cp_sign_west(Config.get_prio_sign_dist(1)))
                    ).to_schema(),
                )
            )
            others.append(
                schema.trafficSign(
                    type=type_map_opposite[self.rule],
                    orientation=self.transform.get_angle() - 0.5 * math.pi + self._alpha,
                    centerPoint=(
                        self.transform
                        * Point(self.cp_sign_east(Config.get_prio_sign_dist(1)))
                    ).to_schema(),
                )
            )

        for obstacle in self.obstacles:
            others.extend(obstacle.export())
        export = Export(southRight, southLeft, others)
        export.lanelet_pairs.append((turn_lines_right, turn_lines_left))
        export.lanelet_pairs.append(last_pair)
        return export
