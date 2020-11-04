import math
import os
from typing import List

from gi import require_version
from gi.repository import Rsvg  # noqa: 402

import simulation.utils.road.renderer.utils as utils  # no
from simulation.utils.geometry import Line, Point, Polygon, Vector
from simulation.utils.road.config import Config
from simulation.utils.road.sections import SurfaceMarking
from simulation.utils.road.sections.road_section import MarkedLine, RoadSection

require_version("Rsvg", "2.0")


def draw(ctx, surface_marking: SurfaceMarking):
    ctx.save()

    if (
        surface_marking.kind == SurfaceMarking.LEFT_TURN_MARKING
        or surface_marking.kind == SurfaceMarking.RIGHT_TURN_MARKING
    ):
        ctx.translate(surface_marking.center.x, surface_marking.center.y)
        ctx.rotate(surface_marking.orientation)
        # Translate to left upper corner of svg
        ctx.translate(
            surface_marking.depth / 2,
            -surface_marking.width / 2,
        )
        # Then rotate by 90 degrees because svgs have the wrong direction!
        ctx.rotate(math.pi / 2)
        image_file = os.path.join(
            os.environ.get("KITCAR_REPO_PATH"),
            "kitcar-gazebo-simulation",
            "simulation",
            "models",
            "meshes",
            "Fahrbahnmarkierung_Pfeil_"
            + (
                "L" if surface_marking.kind != SurfaceMarking.LEFT_TURN_MARKING else "R"
            )  # turn arrows are mirrored in rendering process
            + ".svg",
        )
        svg = Rsvg.Handle().new_from_file(image_file)
        ctx.scale(0.001, 0.001)
        svg.render_cairo(ctx)

    if (
        surface_marking.kind == SurfaceMarking.GIVE_WAY_LINE
        or surface_marking.kind == SurfaceMarking.STOP_LINE
    ):
        ctx.translate(surface_marking.center.x, surface_marking.center.y)
        ctx.rotate(surface_marking.orientation)

        v = 0.5 * Vector(0, surface_marking.width)
        line = MarkedLine(
            [-1 * v, v],
            style=(
                RoadSection.DASHED_LINE_MARKING
                if surface_marking.kind == SurfaceMarking.GIVE_WAY_LINE
                else RoadSection.SOLID_LINE_MARKING
            ),
        )

        utils.draw_line(
            ctx,
            line,
            line_width=0.04,
            dash_length=0.08,
            dash_gap=0.06,
        )
    if surface_marking.kind == SurfaceMarking.START_LINE:
        draw_start_lane(ctx, surface_marking.frame)
    if surface_marking.kind == SurfaceMarking.ZEBRA_CROSSING:
        draw_zebra_crossing(ctx, surface_marking.frame)
    if surface_marking.kind == SurfaceMarking.PARKING_SPOT_X:
        draw_parking_spot_x(ctx, surface_marking.frame)
    if surface_marking.kind == SurfaceMarking.BLOCKED_AREA:
        draw_blocked_area(ctx, surface_marking.frame)
    if surface_marking.kind == SurfaceMarking.ZEBRA_LINES:
        draw_crossing_lines(ctx, surface_marking.frame)
    if surface_marking.kind == SurfaceMarking.TRAFFIC_ISLAND_BLOCKED:
        draw_traffic_island_blocked(ctx, surface_marking.frame)

    ctx.restore()


def draw_start_lane(ctx, frame: Polygon):
    """Draw the checkerboard pattern to mark the beginning of a parking area in the given
    frame.

    Args:
        frame: Frame of the start lane.
            **Points of the frame must be given in the right order!**
            first point : start on left line
            second point: end on left line
            third point : end on right line
            fourth point: start on right line
    """
    TILE_LENGTH = 0.02

    points = frame.get_points()
    left = Line([points[0], points[1]])
    right = Line([points[3], points[2]])
    for i in range(3):
        utils.draw_line(
            ctx,
            MarkedLine(
                [
                    left.interpolate(TILE_LENGTH * (i + 0.5)),
                    right.interpolate(TILE_LENGTH * (i + 0.5)),
                ],
                style=RoadSection.DASHED_LINE_MARKING,
                prev_length=(i % 2 + 0.5) * TILE_LENGTH,
            ),
            dash_length=TILE_LENGTH,
        )


def draw_blocked_area(ctx, frame: Polygon):
    """Draw a blocked area in the given frame.

    Args:
        frame: Frame of the blocked area.
            **Points of the frame must be given in the right order!**
            first point : start on right line
            second point: left of first point, towards middle line
            third point : left of fourth point, towards middle line
            fourth point: end on right line

            Line between second and third point has to be parallel to middle/right line.
    """

    points = frame.get_points()
    left = Line([points[1], points[2], points[3]])
    v = Vector(Vector(points[0]) - Vector(points[3]))
    start = Vector(points[3])
    STRIPES_ANGLE = math.radians(27)
    STRIPES_GAP = 0.15
    draw_blocked_stripes(ctx, v, start, left, points, STRIPES_ANGLE, STRIPES_GAP)


def draw_zebra_crossing(
    ctx, frame: Polygon, stripe_width: float = 0.04, offset: float = 0.02
):
    """Draw a zebra crossing in the given frame.

    Args:
        frame: Frame of the zebra crossing.
            **Points of the frame must be given in the right order!**
            first point : start on left line
            second point: end on left line
            third point : end on right line
            fourth point: start on right line
    """
    points = frame.get_points()
    left = Line([points[0], points[3]])
    right = Line([points[1], points[2]])
    flag = True
    min_length = min(left.length, right.length)
    x = offset
    while x < min_length:
        l, r = left.interpolate(x), right.interpolate(x)
        if flag:
            ctx.move_to(l.x, l.y)
            ctx.line_to(r.x, r.y)
            flag = False
        else:
            ctx.line_to(r.x, r.y)
            ctx.line_to(l.x, l.y)
            ctx.close_path()
            ctx.fill()
            flag = True

        x += stripe_width


def draw_crossing_lines(ctx, frame: Polygon):
    """Draw a crossing area for pedestrian, which is only marked by dashed lines, in the
    given frame.

    The dashed lines are perpendicular to the road.

    Args:
        frame: Frame of the crossing area.
            **Points of the frame must be given in the right order!**
            first point : start on left line
            second point: end on left line
            third point : end on right line
            fourth point: start on right line
    """
    points = frame.get_points()
    dash_length = 0.04
    utils.draw_line(
        ctx,
        MarkedLine(
            [points[0], points[3]],
            style=RoadSection.DASHED_LINE_MARKING,
        ),
        dash_length=dash_length,
    )
    utils.draw_line(
        ctx,
        MarkedLine(
            [points[1], points[2]],
            style=RoadSection.DASHED_LINE_MARKING,
        ),
        dash_length=dash_length,
    )


def draw_traffic_island_blocked(ctx, frame: Polygon):
    """Draw a blocked area, which splits the two lanes of the traffic island, in the given
    frame.

    Args:
        frame: Frame of the blocked area.
            **Points of the frame must be given in the right order!**
            first half of points: left border
            second half: right border
    """
    points = frame.get_points()
    left = Line(points[: len(points) // 2])
    v = Vector(Vector(points[-2]) - Vector(points[0]))
    start = Vector(points[0])
    STRIPES_ANGLE = math.radians(90 - 27)
    STRIPES_GAP = 0.1
    draw_blocked_stripes(ctx, v, start, left, points, STRIPES_ANGLE, STRIPES_GAP)


def draw_parking_spot_x(ctx, frame: Polygon):
    """Draw two crossing lines (X) in the given frame to represent a blocked spot.

    Args:
        frame: Frame of the parking spot.
            **Points of the frame must be given in the right order!**
            first point : left lower corner of parking spot
            second point: left upper corner
            third point : right upper corner
            fourth point: right lower corner
    """
    points = frame.get_points()
    utils.draw_line(
        ctx, MarkedLine([points[0], points[2]], style=RoadSection.SOLID_LINE_MARKING)
    )
    utils.draw_line(
        ctx, MarkedLine([points[1], points[3]], style=RoadSection.SOLID_LINE_MARKING)
    )


def draw_blocked_stripes(
    ctx, v: Vector, start: Point, line: Line, points: List[Point], angle: float, gap: float
):
    """Draw white stripes onto the ground.

    White stripes are e.g. used by to signal areas on the ground
    where the car is not allowed to drive.

    Args:
        v: Vector along the line where the stripes start points are located.
        start: Start point on the line where the stripes start points are located.
        line: End points of the stripes are on this line.
        points: List of points of the polygon frame.
        angle: Angle of the stripes.
        gap: Gap between the stripes.
    """
    ctx.save()

    v = gap / abs(v) * v
    stripe = v.rotated(math.pi + angle)
    # vetcor in direction of stripes
    stripe = 2.5 * Config.road_width / abs(stripe) * stripe

    for point in points:
        ctx.line_to(point.x, point.y)
    ctx.stroke_preserve()
    ctx.clip()

    ctx.set_line_width(0.02)
    while True:
        start += v
        p = Point(start + stripe)
        end = line.intersection(Line([start, p]))
        if end.is_empty:
            break
        ctx.move_to(end.x, end.y)
        ctx.line_to(start.x, start.y)
        ctx.stroke()
    ctx.restore()
