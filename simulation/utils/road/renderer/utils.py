from simulation.utils.geometry import Polygon
from simulation.utils.road.sections.road_section import RoadSection, MarkedLine

from typing import List


def draw_line(
    ctx,
    marked_line: MarkedLine,
    line_width: float = 0.02,
    dash_length: float = 0.2,
    dash_gap: float = None,
):
    if marked_line.style is RoadSection.MISSING_LINE_MARKING:
        return

    ctx.set_source_rgb(1, 1, 1)
    ctx.set_line_width(line_width)

    if marked_line.style == RoadSection.DASHED_LINE_MARKING:
        if dash_gap is None:
            dash_gap = dash_length
        ctx.set_dash(
            [dash_length, dash_gap], marked_line.prev_length % (dash_length + dash_gap),
        )
    else:
        ctx.set_dash([])

    if marked_line.length == 0:
        return

    points = marked_line.get_points()

    ctx.move_to(points[0].x, points[0].y)
    for p in points[1:]:
        ctx.line_to(p.x, p.y)
    ctx.stroke()


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
