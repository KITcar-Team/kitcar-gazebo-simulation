from simulation.utils.road.sections import SurfaceMarking
import os

import simulation.utils.road.renderer.utils as utils  # no
from simulation.utils.road.sections.road_section import RoadSection, MarkedLine
from simulation.utils.geometry import Vector, Line, Polygon

from gi import require_version

require_version("Rsvg", "2.0")
from gi.repository import Rsvg  # noqa: 402


def draw(ctx, surface_marking: SurfaceMarking):
    ctx.save()

    if (
        surface_marking.kind == SurfaceMarking.LEFT_TURN_MARKING
        or surface_marking.kind == SurfaceMarking.RIGHT_TURN_MARKING
    ):
        ctx.translate(surface_marking.center.x, surface_marking.center.y)
        ctx.rotate(surface_marking.orientation)
        image_file = os.path.join(
            os.environ.get("KITCAR_REPO_PATH"),
            "kitcar-gazebo-simulation",
            "simulation",
            "models",
            "meshes",
            f"Fahrbahnmarkierung_Pfeil_"
            + (
                "L"
                if surface_marking.kind != SurfaceMarking.LEFT_TURN_MARKING  # Weird
                else "R"
            )
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

        v = 0.5 * Vector(surface_marking.width, 0)
        line = MarkedLine(
            [-1 * v, v],
            style=(
                RoadSection.DASHED_LINE_MARKING
                if surface_marking.kind == SurfaceMarking.GIVE_WAY_LINE
                else RoadSection.SOLID_LINE_MARKING
            ),
        )

        utils.draw_line(
            ctx, line, line_width=0.04, dash_length=0.08, dash_gap=0.06,
        )
    if surface_marking.kind == SurfaceMarking.START_LINE:
        draw_start_lane(ctx, surface_marking.frame)
    if surface_marking.kind == SurfaceMarking.ZEBRA_CROSSING:
        draw_zebra_crossing(ctx, surface_marking.frame)
    if surface_marking.kind == SurfaceMarking.PARKING_SPOT_X:
        draw_parking_spot_x(ctx, surface_marking.frame)

    ctx.restore()


def draw_start_lane(ctx, frame: Polygon):
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


def draw_zebra_crossing(
    ctx, frame: Polygon, stripe_width: float = 0.04, offset: float = 0.02
):
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


def draw_parking_spot_x(ctx, frame: Polygon):
    points = frame.get_points()
    utils.draw_line(
        ctx, MarkedLine([points[0], points[2]], style=RoadSection.SOLID_LINE_MARKING)
    )
    utils.draw_line(
        ctx, MarkedLine([points[1], points[3]], style=RoadSection.SOLID_LINE_MARKING)
    )
