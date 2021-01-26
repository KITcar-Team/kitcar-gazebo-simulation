from typing import List

from simulation.utils.geometry import Polygon
from simulation.utils.road.sections.road_section import MarkedLine, RoadSection


def draw_line(
    ctx,
    marked_line: MarkedLine,
    line_width: float = 0.02,
    dash_length: float = 0.2,
    dash_gap: float = None,
    double_line_gap: float = 0.02,
):
    if marked_line.style is RoadSection.MISSING_LINE_MARKING or marked_line.length == 0:
        return

    ctx.set_source_rgb(1, 1, 1)
    ctx.set_line_width(line_width)

    if marked_line.style == RoadSection.DOUBLE_SOLID_LINE_MARKING:
        _draw_double_line(
            ctx,
            line=marked_line,
            style_left=RoadSection.SOLID_LINE_MARKING,
            style_right=RoadSection.SOLID_LINE_MARKING,
            line_width=line_width,
            double_line_gap=double_line_gap,
        )
        return
    elif marked_line.style == RoadSection.DOUBLE_DASHED_LINE_MARKING:
        _draw_double_line(
            ctx,
            line=marked_line,
            style_left=RoadSection.DASHED_LINE_MARKING,
            style_right=RoadSection.DASHED_LINE_MARKING,
            line_width=line_width,
            double_line_gap=double_line_gap,
        )
        return
    elif marked_line.style == RoadSection.SOLID_DASHED_LINE_MARKING:
        _draw_double_line(
            ctx,
            line=marked_line,
            style_left=RoadSection.SOLID_LINE_MARKING,
            style_right=RoadSection.DASHED_LINE_MARKING,
            line_width=line_width,
            double_line_gap=double_line_gap,
        )
        return
    elif marked_line.style == RoadSection.DASHED_SOLID_LINE_MARKING:
        _draw_double_line(
            ctx,
            line=marked_line,
            style_left=RoadSection.DASHED_LINE_MARKING,
            style_right=RoadSection.SOLID_LINE_MARKING,
            line_width=line_width,
            double_line_gap=double_line_gap,
        )
        return
    elif marked_line.style == RoadSection.DASHED_LINE_MARKING:
        if dash_gap is None:
            dash_gap = dash_length
        ctx.set_dash(
            [dash_length, dash_gap],
            marked_line.prev_length % (dash_length + dash_gap),
        )
    else:  # implicit RoadSection.SOLID_LINE_MARKING
        ctx.set_dash([])

    points = marked_line.get_points()
    ctx.move_to(points[0].x, points[0].y)
    for p in points[1:]:
        ctx.line_to(p.x, p.y)
    ctx.stroke()


def _draw_double_line(
    ctx,
    line: MarkedLine,
    style_left: str,
    style_right: str,
    line_width: float,
    double_line_gap: float,
):
    left_line = line.parallel_offset(double_line_gap / 2 + line_width / 2, "left")
    right_line = line.parallel_offset(double_line_gap / 2 + line_width / 2, "right")
    marked_left_line = MarkedLine.from_line(left_line, style_left, line.prev_length)
    marked_right_line = MarkedLine.from_line(right_line, style_right, line.prev_length)
    draw_line(ctx, marked_left_line)
    draw_line(ctx, marked_right_line)


def draw_polygon(ctx, polygon: Polygon, lines: List[bool] = None):

    points = polygon.get_points()
    if lines is None:
        lines = [True for _ in range(len(points) - 1)]
    assert len(points) == len(lines) + 1

    ctx.save()
    ctx.set_line_width(0.02)

    ctx.move_to(points[0].x, points[0].y)
    for i, p in enumerate(points[1:]):
        if lines[i]:
            ctx.line_to(p.x, p.y)
            ctx.stroke()
        ctx.move_to(p.x, p.y)

    ctx.restore()
