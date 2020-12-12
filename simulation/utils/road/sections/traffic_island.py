"""TrafficIsland."""

import math
from dataclasses import dataclass
from typing import List

import simulation.utils.road.sections.type as road_section_type
from simulation.utils.geometry import Line, Point, Polygon, Vector
from simulation.utils.road.config import Config
from simulation.utils.road.sections import SurfaceMarking, SurfaceMarkingPoly, TrafficSign
from simulation.utils.road.sections.bezier_curve import add_quad_bezier_points
from simulation.utils.road.sections.road_section import MarkedLine, RoadSection


@dataclass
class TrafficIsland(RoadSection):
    """Road section representing an traffic island.

    Args:
        island_width (float) = 0.3: width of the island in the middle
        zebra_length (float) = 0.45: length of zebra section on the island
        curve_area_length (float) = 0.8: length of curve area section
        curvature (float) = 0.5: amount of curvature ranging from 0 to 1
        zebra_marking_type (int) = TrafficIsland.LINES: marking on the middle of the island
    """

    TYPE = road_section_type.TRAFFIC_ISLAND

    LINES = 0
    """Possible value for :attr:`zebra_marking_type`. Show lines on the island."""
    ZEBRA = 1
    """Possible value for :attr:`zebra_marking_type`. Show zebra crossing on the island."""

    island_width: float = 0.3
    """Width of island in the middle."""
    zebra_length: float = 0.45
    """Length of the zebra crossing area."""
    curve_area_length: float = 0.8
    """Length of the bezier curve area at start and end of island."""
    curvature: float = 0.4
    """Define where along the curve area the control points are located."""
    zebra_marking_type: int = ZEBRA
    """Type of zebra marking type. Can be LINES or ZEBRA."""
    _sign_distance: float = 0.30
    """Distance of the directions signs from the mid island part."""

    def __post_init__(self):
        super().__post_init__()
        self.p_offset = self.curve_area_length * self.curvature
        """Offset for the bezier control points of the curve area."""

    @property
    def length(self) -> float:
        """Length of the entire section."""
        return self.curve_area_length * 2 + self.zebra_length

    @property
    def middle_start(self) -> Vector:
        return Vector(0, 0)

    @property
    def middle_r_zebra_start(self) -> Vector:
        return self.middle_start + Vector(self.curve_area_length, -self.island_width / 2)

    @property
    def middle_r_zebra_end(self) -> Vector:
        return self.middle_r_zebra_start + Vector(self.zebra_length, 0)

    @property
    def middle_end(self) -> Vector:
        return self.middle_r_zebra_end + Vector(
            self.curve_area_length, self.island_width / 2
        )

    @property
    def right_zebra_start(self) -> Vector:
        return self.middle_start + Vector(
            self.curve_area_length, -Config.road_width - self.island_width / 2
        )

    @property
    def right_zebra_end(self) -> Vector:
        return self.right_zebra_start + Vector(self.zebra_length, 0)

    @property
    def left_zebra_start(self) -> Vector:
        return self.middle_start + Vector(
            self.curve_area_length, Config.road_width + self.island_width / 2
        )

    @property
    def left_zebra_end(self) -> Vector:
        return self.left_zebra_start + Vector(self.zebra_length, 0)

    @property
    def middle_l_zebra_start(self) -> Vector:
        return self.middle_start + Vector(self.curve_area_length, self.island_width / 2)

    @property
    def middle_l_zebra_end(self) -> Vector:
        return self.middle_l_zebra_start + Vector(self.zebra_length, 0)

    @property
    def bezier_points_mid_r_start(self) -> List[Point]:
        return add_quad_bezier_points(
            self.middle_start.to_numpy(),
            (self.middle_start + Vector(self.p_offset, 0)).to_numpy(),
            (self.middle_r_zebra_start - Vector(self.p_offset, 0)).to_numpy(),
            self.middle_r_zebra_start.to_numpy(),
        )

    @property
    def bezier_points_mid_r_end(self) -> List[Point]:
        return add_quad_bezier_points(
            self.middle_r_zebra_end.to_numpy(),
            (self.middle_r_zebra_end + Vector(self.p_offset, 0)).to_numpy(),
            (self.middle_end - Vector(self.p_offset, 0)).to_numpy(),
            self.middle_end.to_numpy(),
        )

    @property
    def bezier_points_mid_l_start(self) -> List[Point]:
        return add_quad_bezier_points(
            self.middle_start.to_numpy(),
            (self.middle_start + Vector(self.p_offset, 0)).to_numpy(),
            (self.middle_l_zebra_start - Vector(self.p_offset, 0)).to_numpy(),
            self.middle_l_zebra_start.to_numpy(),
        )

    @property
    def bezier_points_mid_l_end(self) -> List[Point]:
        return add_quad_bezier_points(
            self.middle_l_zebra_end.to_numpy(),
            (self.middle_l_zebra_end + Vector(self.p_offset, 0)).to_numpy(),
            (self.middle_end - Vector(self.p_offset, 0)).to_numpy(),
            self.middle_end.to_numpy(),
        )

    @property
    def middle_line_r(self) -> Line:
        """Line: Middle line on the right side of the traffic island."""
        return (
            self.transform
            * Line(
                [
                    self.middle_start,
                    *self.bezier_points_mid_r_start,
                    self.middle_r_zebra_start,
                    self.middle_r_zebra_end,
                    *self.bezier_points_mid_r_end,
                    self.middle_end,
                ]
            ).simplify()
        )
        """The simplification is necessary to be able to draw the blocked areas of the island.

        If the line is not simplified the result of the intersection process
        which is used to draw the blocked stripes is a MultiPoints.
        """

    @property
    def middle_line_l(self) -> Line:
        """Line: Middle line on the left side of the traffic island."""
        return (
            self.transform
            * Line(
                [
                    self.middle_start,
                    *self.bezier_points_mid_l_start,
                    self.middle_l_zebra_start,
                    self.middle_l_zebra_end,
                    *self.bezier_points_mid_l_end,
                    self.middle_end,
                ]
            ).simplify()
        )

    @property
    def middle_line(self) -> Line:
        """Line: Middle line of the road section.
        Here it is the left middle line.
        """
        return self.middle_line_r

    @property
    def right_line(self) -> Line:
        """Line: Right line of the road section."""
        return self.middle_line_r.parallel_offset(Config.road_width, "right")

    @property
    def left_line(self) -> Line:
        """Line: Left line of the road section."""
        return self.middle_line_l.parallel_offset(Config.road_width, "left")

    @property
    def lines(self) -> List[MarkedLine]:
        """List[MarkedLine]: All road lines with their marking type."""
        lines = []
        lines.append(
            MarkedLine.from_line(self.left_line, self.left_line_marking, self.prev_length)
        )
        lines.append(
            MarkedLine.from_line(
                self.middle_line, self.SOLID_LINE_MARKING, self.prev_length
            )
        )
        lines.append(
            MarkedLine.from_line(self.right_line, self.right_line_marking, self.prev_length)
        )
        lines.append(
            MarkedLine.from_line(
                self.middle_line_l, self.SOLID_LINE_MARKING, self.prev_length
            )
        )
        return lines

    @property
    def traffic_signs(self) -> List[TrafficSign]:
        """List[TrafficSign]: All traffic signs within this section of the road."""
        traffic_signs = RoadSection.traffic_signs.fget(self)
        traffic_sign_start_point = self.transform * Point(
            self.curve_area_length - self._sign_distance, 0
        )
        traffic_signs.append(
            TrafficSign(
                kind=TrafficSign.PASS_RIGHT,
                center=traffic_sign_start_point,
                angle=self.transform.get_angle(),
                normalize_x=False,
            )
        )
        traffic_sign_end_point = self.transform * Point(
            self.length - self.curve_area_length + self._sign_distance, 0
        )
        traffic_signs.append(
            TrafficSign(
                kind=TrafficSign.PASS_RIGHT,
                center=traffic_sign_end_point,
                angle=self.transform.get_angle() + math.pi,
                normalize_x=False,
            )
        )
        if self.zebra_marking_type == self.ZEBRA:
            vector = Vector(
                self.right_line.interpolate(
                    self.right_line.project(traffic_sign_start_point)
                )
            )
            point = vector + Vector(0, -0.1).rotated(self.transform.get_angle())
            traffic_signs.append(
                TrafficSign(
                    kind=TrafficSign.ZEBRA_CROSSING,
                    center=point,
                    angle=self.transform.get_angle(),
                    normalize_x=False,
                )
            )
        return traffic_signs

    @traffic_signs.setter
    def traffic_signs(self, signs: List[TrafficSign]):
        self._traffic_signs = signs

    @property
    def surface_markings(self) -> List[SurfaceMarking]:
        """List[SurfaceMarking]: All surface markings within this section of the road."""
        sf_marks = RoadSection.surface_markings.fget(self)

        right_poly = self.transform * Polygon(
            [
                self.right_zebra_start,
                self.right_zebra_end,
                self.middle_r_zebra_end,
                self.middle_r_zebra_start,
            ]
        )
        left_poly = self.transform * Polygon(
            [
                self.left_zebra_start,
                self.left_zebra_end,
                self.middle_l_zebra_end,
                self.middle_l_zebra_start,
            ]
        )
        if self.zebra_marking_type == self.LINES:
            kind = SurfaceMarkingPoly.ZEBRA_LINES
        else:
            kind = SurfaceMarkingPoly.ZEBRA_CROSSING

        sf_marks.append(SurfaceMarkingPoly(frame=right_poly, kind=kind, normalize_x=False))
        sf_marks.append(SurfaceMarkingPoly(frame=left_poly, kind=kind, normalize_x=False))
        blocked_start_poly = self.transform * Polygon(
            [
                self.middle_l_zebra_start,
                *reversed(self.bezier_points_mid_l_start),
                self.middle_start,
                *self.bezier_points_mid_r_start,
                self.middle_r_zebra_start,
            ]
        )
        blocked_end_poly = self.transform * Polygon(
            [
                self.middle_r_zebra_end,
                *self.bezier_points_mid_r_end,
                self.middle_end,
                *reversed(self.bezier_points_mid_l_end),
                self.middle_l_zebra_end,
            ]
        )
        sf_marks.append(
            SurfaceMarkingPoly(
                frame=blocked_start_poly,
                kind=SurfaceMarkingPoly.TRAFFIC_ISLAND_BLOCKED,
                normalize_x=False,
            )
        )
        sf_marks.append(
            SurfaceMarkingPoly(
                frame=blocked_end_poly,
                kind=SurfaceMarkingPoly.TRAFFIC_ISLAND_BLOCKED,
                normalize_x=False,
            )
        )
        return sf_marks

    @surface_markings.setter
    def surface_markings(self, markings: List[SurfaceMarking]):
        self._surface_markings = markings
