import cairo

# have to use pycairo instead of cairocffi
# as Rsvg bindings don't work with the latter
# import cairocffi as cairo
import math
import road.utils
from os import path
import os
import hashlib
from tqdm import tqdm
import numpy as np
from enum import Enum
from collections import namedtuple
import gi

gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg
import shutil
import sys

from geometry import Line, Point

sys.path.append(
    os.path.join(
        os.environ.get("KITCAR_REPO_PATH"),
        "kitcar-gazebo-simulation",
        "utils",
        "machine_learning",
    )
)
import machine_learning.color_classes as color_classes

PIXEL_PER_UNIT = 500
PADDING = 3

NUMBER_COLORS = 5


class MarkerImage(Enum):
    TURN_LEFT = "road/renderer/street_markings/" "Fahrbahnmarkierung_Pfeil_L.svg"
    TURN_RIGHT = (
        "road/renderer/street_markings/" "Fahrbahnmarkierung_Pfeil_R.svg"
    )


StreetMarking = namedtuple("StreetMarking", ["marker_image", "marker_text", "crossed"])


ROADMARKING_TYPE_TO_VISUAL = {
    "10_zone_beginn": StreetMarking(marker_image=None, marker_text="10", crossed=False),
    "20_zone_beginn": StreetMarking(marker_image=None, marker_text="20", crossed=False),
    "stvo-274.1": StreetMarking(marker_image=None, marker_text="30", crossed=False),
    "40_zone_beginn": StreetMarking(marker_image=None, marker_text="40", crossed=False),
    "50_zone_beginn": StreetMarking(marker_image=None, marker_text="50", crossed=False),
    "60_zone_beginn": StreetMarking(marker_image=None, marker_text="60", crossed=False),
    "70_zone_beginn": StreetMarking(marker_image=None, marker_text="70", crossed=False),
    "80_zone_beginn": StreetMarking(marker_image=None, marker_text="80", crossed=False),
    "90_zone_beginn": StreetMarking(marker_image=None, marker_text="90", crossed=False),
    "ende_10_zone": StreetMarking(marker_image=None, marker_text="10", crossed=True),
    "ende_20_zone": StreetMarking(marker_image=None, marker_text="20", crossed=True),
    "stvo-274.2": StreetMarking(marker_image=None, marker_text="30", crossed=True),
    "ende_40_zone": StreetMarking(marker_image=None, marker_text="40", crossed=True),
    "ende_50_zone": StreetMarking(marker_image=None, marker_text="50", crossed=True),
    "ende_60_zone": StreetMarking(marker_image=None, marker_text="60", crossed=True),
    "ende_70_zone": StreetMarking(marker_image=None, marker_text="70", crossed=True),
    "ende_80_zone": StreetMarking(marker_image=None, marker_text="80", crossed=True),
    "ende_90_zone": StreetMarking(marker_image=None, marker_text="90", crossed=True),
    "turn_left": StreetMarking(
        marker_image=MarkerImage.TURN_LEFT, marker_text=None, crossed=False
    ),
    "turn_right": StreetMarking(
        marker_image=MarkerImage.TURN_RIGHT, marker_text=None, crossed=False
    ),
}

def draw_stop_line(ctx, lanelet):
    ctx.save()
    left = Line([Point(point.x, point.y) for point in lanelet.leftBoundary.point])
    right = Line([Point(point.x, point.y) for point in lanelet.rightBoundary.point])
    lineWidth = 0.04
    segmentLength = 0.08
    segmentGap = 0.06
    if lanelet.stopLine == "dashed":
        if lanelet.stopLineAttributes:
            lineWidth = lanelet.stopLineAttributes.lineWidth
            segmentLength = lanelet.stopLineAttributes.segmentLength
            segmentGap = lanelet.stopLineAttributes.segmentGap
        ctx.set_dash([segmentLength, segmentGap])
    else:
        ctx.set_dash([])
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    ctx.set_line_width(lineWidth)
    if(left.length < right.length):
        last_left = left.get_points()[-1]
        point_right = right.interpolate(right.project(last_left))
        ctx.move_to(last_left.x, last_left.y)
        ctx.line_to(point_right.x, point_right.y)
    else:
        last_right = right.get_points()[-1]
        point_left = left.interpolate(left.project(last_right))
        ctx.move_to(last_right.x, last_right.y)
        ctx.line_to(point_left.x, point_left.y)
    ctx.stroke()
    ctx.restore()


def draw_rectangle(ctx, rectangle):
    ctx.save()
    ctx.translate(rectangle.centerPoint.x, rectangle.centerPoint.y)
    ctx.rotate(-rectangle.orientation)
    ctx.rectangle(
        -rectangle.length / 2, -rectangle.width / 2, rectangle.length, rectangle.width,
    )
    ctx.fill()
    ctx.restore()


def draw_circle(ctx, circle):
    ctx.arc(
        circle.centerPoint.x, circle.centerPoint.y, circle.radius, 0, 2 * math.pi,
    )
    ctx.fill()


def draw_polygon(ctx, polygon):
    ctx.move_to(polygon.point[0].x, polygon.point[1].y)
    for point in polygon.point[1:]:
        ctx.line_to(point.x, point.y)
    ctx.fill()


def draw_shape(ctx, shape):
    for rect in shape.rectangle:
        draw_rectangle(ctx, rect)
    for circ in shape.circle:
        draw_circle(ctx, circ)
    for poly in shape.polygon:
        draw_polygon(ctx, poly)


def draw_island_junction(ctx, island):
    ctx.save()
    ctx.set_dash([])
    ctx.set_line_width(0.02)
    for i in range(0, len(island.point), 2):
        ctx.move_to(island.point[i].x, island.point[i].y)
        ctx.line_to(island.point[i + 1].x, island.point[i + 1].y)
    ctx.stroke()
    ctx.restore()


def draw_road_marking(ctx, marking, generator_dir):
    """Draw road markings into context

    @param ctx Surface context

    @param marking Specifies marking type

    @param generator_dir Directory which contains road_generation
    """

    marking_visual = ROADMARKING_TYPE_TO_VISUAL[marking.type]
    if marking_visual.marker_text:
        ctx.save()
        ctx.set_dash([])
        font = "DIN 1451 Std"
        font_size = 0.4
        text = "30"
        font_args = [cairo.FONT_SLANT_NORMAL]
        ctx.translate(
            marking.centerPoint.x,  # -0.145*math.cos(marking.orientation)
            marking.centerPoint.y,  # -0.145*math.sin(marking.orientation)
        )
        ctx.rotate(marking.orientation)
        # mirror text
        ctx.transform(cairo.Matrix(1.0, 0, 0, -1, 0, 0))
        ctx.translate(-0.145, 0.29)
        ctx.select_font_face(font, *font_args)
        ctx.set_font_size(font_size)
        ctx.text_path(marking_visual.marker_text)
        ctx.set_line_width(0.01)
        (
            x_bearing,
            y_bearing,
            text_width,
            text_height,
            x_advance,
            y_advance,
        ) = ctx.text_extents(text)
        ctx.fill_preserve()
        ctx.stroke()
        ctx.restore()
    if marking_visual.crossed:
        ctx.save()
        ctx.move_to(
            marking.centerPoint.x + 0.145 * math.cos(marking.orientation),
            marking.centerPoint.y + 0.145 * math.sin(marking.orientation),
        )
        ctx.line_to(
            marking.centerPoint.x
            + 0.145 * math.cos(marking.orientation)
            - text_height * math.cos(marking.orientation)
            + text_width * math.sin(marking.orientation),
            marking.centerPoint.y
            + 0.145 * math.sin(marking.orientation)
            - text_height * math.sin(marking.orientation)
            - text_width * math.cos(marking.orientation),
        )
        ctx.move_to(
            (marking.centerPoint.x + (0.145 - text_height) * math.cos(marking.orientation)),
            (marking.centerPoint.y + (0.145 - text_height) * math.sin(marking.orientation)),
        )
        ctx.line_to(
            (
                marking.centerPoint.x
                + 0.145 * math.cos(marking.orientation)
                + text_width * math.sin(marking.orientation)
            ),
            (
                marking.centerPoint.y
                + 0.145 * math.sin(marking.orientation)
                - text_width * math.cos(marking.orientation)
            ),
        )
        ctx.set_line_width(0.05)
        ctx.stroke()
        ctx.restore()

    if marking_visual.marker_image:
        ctx.save()
        handle = Rsvg.Handle()
        svg = handle.new_from_file(
            path.join(generator_dir, marking_visual.marker_image.value)
        )
        ctx.translate(marking.centerPoint.x, marking.centerPoint.y)
        ctx.rotate(marking.orientation)
        ctx.scale(0.001, 0.001)
        svg.render_cairo(ctx)
        ctx.restore()


def draw_stripes_rect(ctx, rectangle):
    ctx.save()
    ctx.translate(rectangle.centerPoint.x, rectangle.centerPoint.y)
    ctx.rotate(-rectangle.orientation)

    ctx.set_line_width(0.02)
    sheering = rectangle.width / 2
    ctx.move_to(-rectangle.length / 2, -rectangle.width / 2)
    ctx.line_to(rectangle.length / 2, -rectangle.width / 2)
    ctx.line_to(rectangle.length / 2 - sheering, rectangle.width / 2)
    ctx.line_to(-rectangle.length / 2 + sheering, rectangle.width / 2)
    ctx.close_path()
    ctx.clip_preserve()
    ctx.stroke()

    start_x = -rectangle.length / 2 - rectangle.width
    end_x = rectangle.length / 2
    y_bottom = -rectangle.width / 2
    y_top = rectangle.width / 2
    ctx.set_line_width(0.02)
    for x in np.arange(start_x, end_x, 0.08):
        ctx.move_to(x, y_bottom)
        ctx.line_to(x + rectangle.width, y_top)
    ctx.stroke()
    ctx.restore()


def draw_zebra_crossing(ctx, lanelet):
    left = boundary_to_equi_distant(lanelet.leftBoundary, 0.04, 0.02)
    right = boundary_to_equi_distant(lanelet.rightBoundary, 0.04, 0.02)
    flag = True
    ctx.save()
    for (l, r) in zip(left, right):
        if flag:
            ctx.move_to(l[0], l[1])
            ctx.line_to(r[0], r[1])
            flag = False
        else:
            ctx.line_to(r[0], r[1])
            ctx.line_to(l[0], l[1])
            ctx.close_path()
            ctx.fill()
            flag = True
    ctx.restore()


def draw_start_lane(ctx, lanelet):
    TILE_LENGTH = 0.02

    left = Line(boundary_to_equi_distant(lanelet.leftBoundary, 0.01, 0.01))
    right = Line(boundary_to_equi_distant(lanelet.rightBoundary, 0.01, 0.01))

    x = TILE_LENGTH

    idx = 0
    while x < left.length + TILE_LENGTH:
        left_lower_border = left.interpolate(x - TILE_LENGTH)
        right_lower_border = right.interpolate(x - TILE_LENGTH)
        # ctx.save()

        left_upper_border = left.interpolate(x)
        right_upper_border = right.interpolate(x)

        cross_lower = Line([right_lower_border, left_lower_border])
        cross_upper = Line([right_upper_border, left_upper_border])

        o = TILE_LENGTH

        idx += 1
        idx %= 2

        flag = idx
        while o <= cross_lower.length:
            r_upper = Point(cross_upper.interpolate(o - TILE_LENGTH))  # From right to left
            r_lower = Point(cross_lower.interpolate(o - TILE_LENGTH))
            l_upper = Point(cross_upper.interpolate(o))  # From right to left
            l_lower = Point(cross_lower.interpolate(o))

            ctx.save()

            # Dont paint
            if flag == 1:
                ctx.move_to(r_lower.x, r_lower.y)
                ctx.line_to(r_upper.x, r_upper.y)
                ctx.line_to(l_upper.x, l_upper.y)
                ctx.line_to(l_lower.x, l_lower.y)
                ctx.close_path()
                ctx.fill()
                flag = 0
            else:
                flag = 1
            o += TILE_LENGTH

            ctx.restore()

        x += TILE_LENGTH


def draw_parking_spot_x(ctx, lanelet):
    ctx.save()
    ctx.move_to(lanelet.leftBoundary.point[0].x, lanelet.leftBoundary.point[0].y)
    ctx.line_to(lanelet.rightBoundary.point[1].x, lanelet.rightBoundary.point[1].y)
    ctx.set_line_width(0.02)
    ctx.stroke()

    ctx.move_to(lanelet.leftBoundary.point[1].x, lanelet.leftBoundary.point[1].y)
    ctx.line_to(lanelet.rightBoundary.point[0].x, lanelet.rightBoundary.point[0].y)
    ctx.set_line_width(0.02)
    ctx.stroke()

    ctx.restore()


def distance_points(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.sqrt(dx * dx + dy * dy)


def boundary_length(boundary):
    length = 0
    for (p1, p2) in zip(boundary.point, boundary.point[1:]):
        length += distance_points(p1, p2)
    return length


def boundary_point_lengths(boundary):
    result = [0]
    len = 0
    for (p1, p2) in zip(boundary.point, boundary.point[1:]):
        len += distance_points(p1, p2)
        result.append(len)
    return result


def boundary_to_equi_distant(boundary, step_width, offset):
    lengths = boundary_point_lengths(boundary)
    x = list(map(lambda p: p.x, boundary.point))
    y = list(map(lambda p: p.y, boundary.point))
    eval_marks = np.arange(offset, lengths[-1], step_width)
    xinterp = np.interp(eval_marks, lengths, x)
    yinterp = np.interp(eval_marks, lengths, y)
    return map(lambda i: (i[0], i[1]), zip(xinterp, yinterp))


def draw_obstacle(ctx, obstacle):
    if obstacle.type == "blockedArea":
        for rect in obstacle.shape.rectangle:
            draw_stripes_rect(ctx, rect)
    # uncomment if you want white boxes under the obstacles
    # else:
    #    draw_shape(ctx, obstacle.shape)


def draw_all_boundaries(ctx, lanelet_list, boundary_name):
    all_ids = [
        lanelet.id
        for lanelet in lanelet_list
        if getattr(lanelet, boundary_name).lineMarking is not None
    ]
    while len(all_ids) > 0:
        current_id = all_ids[0]
        suc = expand_boundary(
            lanelet_list,
            get_lanelet_by_id(lanelet_list, current_id),
            boundary_name,
            "successor",
        )
        pred = expand_boundary(
            lanelet_list,
            get_lanelet_by_id(lanelet_list, current_id),
            boundary_name,
            "predecessor",
        )
        ids_in_run = pred[::-1] + [current_id] + suc

        for id in ids_in_run:
            all_ids.remove(id)

        lanelets = list(map(lambda x: get_lanelet_by_id(lanelet_list, x), ids_in_run))

        ctx.save()
        ctx.set_line_width(0.02)
        line_marking = getattr(lanelets[0], boundary_name).lineMarking
        if line_marking == "parking":
            ctx.set_source_rgb(*color_classes.rgb(color_classes.ColorClass.PARKING))
            ctx.set_dash([])
        elif line_marking == "dashed":
            ctx.set_source_rgb(*color_classes.rgb(color_classes.ColorClass.MIDDLE_LINE))
            ctx.set_dash([0.2, 0.2])
        elif line_marking == "solid":
            ctx.set_source_rgb(*color_classes.rgb(color_classes.ColorClass.SIDE_LINE))
            ctx.set_dash([])
        else:
            continue

        ctx.move_to(
            getattr(lanelets[0], boundary_name).point[0].x,
            getattr(lanelets[0], boundary_name).point[0].y,
        )

        for lanelet in lanelets:
            for p in getattr(lanelet, boundary_name).point:
                ctx.line_to(p.x, p.y)
        ctx.stroke()
        ctx.restore()


def get_lanelet_by_id(lanelet_list, id):
    for lanelet in lanelet_list:
        if lanelet.id == id:
            return lanelet
    return None


def expand_boundary(lanelet_list, lanelet, boundary_name, direction):
    ids = []
    original_line_type = getattr(lanelet, boundary_name).lineMarking
    found = True
    while found:
        found = False
        if getattr(lanelet, direction) is not None:
            for next in getattr(lanelet, direction).lanelet:
                next_lanelet = get_lanelet_by_id(lanelet_list, next.ref)
                if getattr(next_lanelet, boundary_name).lineMarking == original_line_type:
                    lanelet = next_lanelet
                    ids.append(lanelet.id)
                    found = True
                    break
    return ids


def draw(
    doc, generator_dir, road_name, tile_size=4, include_empty=True, segmentation=False,
):
    """Draw Xml tree document

    @param doc Document containing all xml information

    @param road_name name of road

    @param generator_dir Directory which contains road_generation

    @param tile_size Size of quadratic tile measured in meters
    """
    color_classes.SEGMENTATION_ENABLED = segmentation

    tile_size = int(PIXEL_PER_UNIT * tile_size)  # Convert tile size to pixels

    materials_dir = os.path.join(
        os.environ.get("KITCAR_REPO_PATH"),
        "kitcar-gazebo-simulation",
        "simulation",
        "models/env_db",
        "." + road_name,
        "materials",
    )

    # materials_dir = os.path.join(generator_dir,road_name)

    bounding_box = road.utils.get_bounding_box(doc)
    bounding_box.x_min -= PADDING
    bounding_box.y_min -= PADDING
    bounding_box.x_max += PADDING
    bounding_box.y_max += PADDING

    # Adjust bounding box to fit number of tiles and position y=0 in middle
    # of a tile - 0.5 * tile_size/PIXEL_PER_UNIT
    bounding_box.x_min = (
        math.floor(bounding_box.x_min * PIXEL_PER_UNIT / tile_size)
        * tile_size
        / PIXEL_PER_UNIT
    )
    bounding_box.y_min = (
        math.floor(bounding_box.y_min * PIXEL_PER_UNIT / tile_size)
        * tile_size
        / PIXEL_PER_UNIT
        - 0.5 * tile_size / PIXEL_PER_UNIT
    )
    # + 0.5 * tile_size/PIXEL_PER_UNIT
    bounding_box.x_max = (
        math.ceil(bounding_box.x_max * PIXEL_PER_UNIT / tile_size)
        * tile_size
        / PIXEL_PER_UNIT
    )
    bounding_box.y_max = (
        math.ceil(bounding_box.y_max * PIXEL_PER_UNIT / tile_size)
        * tile_size
        / PIXEL_PER_UNIT
        + 0.5 * tile_size / PIXEL_PER_UNIT
    )

    width = math.ceil((bounding_box.x_max - bounding_box.x_min) * PIXEL_PER_UNIT)
    height = math.ceil((bounding_box.y_max - bounding_box.y_min) * PIXEL_PER_UNIT)

    shutil.rmtree(materials_dir, ignore_errors=True)

    width_num = math.ceil(width / tile_size)
    height_num = math.ceil(height / tile_size)

    os.makedirs(path.join(materials_dir, "textures"), exist_ok=True)
    os.makedirs(path.join(materials_dir, "scripts"), exist_ok=True)

    models = []

    for (x, y) in tqdm([(x, y) for x in range(width_num) for y in range(height_num)]):
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, tile_size, tile_size)
        ctx = cairo.Context(surface)

        # fill black
        ctx.set_source_rgb(0, 0, 0)
        ctx.rectangle(0, 0, tile_size, tile_size)
        ctx.fill()

        # Inverse y-axis
        ctx.translate(0, tile_size / 2)
        ctx.scale(1, -1)
        ctx.translate(0, -tile_size / 2)

        ctx.scale(PIXEL_PER_UNIT, PIXEL_PER_UNIT)
        ctx.translate(-bounding_box.x_min, -bounding_box.y_min)
        ctx.translate(-x * tile_size / PIXEL_PER_UNIT, -y * tile_size / PIXEL_PER_UNIT)

        ctx.set_source_rgb(*color_classes.rgb(color_classes.ColorClass.DEFAULT))

        # Get hash of empty tile
        sha_256 = hashlib.sha256()
        sha_256.update(surface.get_data())
        empty_hash = sha_256.hexdigest()

        for lanelet in doc.lanelet:
            if lanelet.stopLine:
                ctx.set_source_rgb(*color_classes.rgb(color_classes.ColorClass.STOP_LINE))
                draw_stop_line(ctx, lanelet)
            if lanelet.type == "zebraCrossing":
                ctx.set_source_rgb(
                    *color_classes.rgb(color_classes.ColorClass.ZEBRA_CROSSING)
                )
                draw_zebra_crossing(ctx, lanelet)

            if lanelet.type == "startLane":
                ctx.set_source_rgb(*color_classes.rgb(color_classes.ColorClass.START_LINE))
                draw_start_lane(ctx, lanelet)

            if lanelet.type == "parking_spot_x":
                ctx.set_source_rgb(
                    *color_classes.rgb(color_classes.ColorClass.PARKING_SPOT_X)
                )
                draw_parking_spot_x(ctx, lanelet)

        draw_all_boundaries(ctx, doc.lanelet, "leftBoundary")

        ctx.set_source_rgb(*color_classes.rgb(color_classes.ColorClass.DEFAULT))

        draw_all_boundaries(ctx, doc.lanelet, "rightBoundary")

        ctx.set_source_rgb(*color_classes.rgb(color_classes.ColorClass.DEFAULT))

        for obstacle in doc.obstacle:
            draw_obstacle(ctx, obstacle)

        for island_junction in doc.islandJunction:
            draw_island_junction(ctx, island_junction)

        for road_marking in doc.roadMarking:
            draw_road_marking(ctx, road_marking, generator_dir)

        sha_256 = hashlib.sha256()
        sha_256.update(surface.get_data())
        hash = sha_256.hexdigest()

        if include_empty or empty_hash != hash:

            texture_file = "tile-{0}.png".format(hash)
            material_file = "tile-{0}.material".format(hash)
            surface.write_to_png(path.join(materials_dir, "textures", texture_file))

            with open(path.join(materials_dir, "scripts", material_file), "w") as file:
                file.write(ground_plane_material("Tile/" + hash, texture_file))

            models.append(
                ground_plane_model(
                    bounding_box.x_min + (x + 0.5) * tile_size / PIXEL_PER_UNIT,
                    bounding_box.y_min + (y + 0.5) * tile_size / PIXEL_PER_UNIT,
                    tile_size / PIXEL_PER_UNIT,
                    "Tile/{0}-{1}".format(x, y),
                    "Tile/" + hash,
                    road_name,
                )
            )

    return models


def ground_plane_material(name, file):
    return """
    material {name}
    {{
        technique
        {{
            pass
            {{
                reflective 1 1 1 1
                texture_unit
                {{
                    texture {file} PF_RGB8
                    filtering anisotropic
                    max_anisotropy 16
                }}
            }}
        }}
    }}
    """.format(
        name=name, file=file
    )


def ground_plane_model(x, y, tile_size, name, material, road_name):
    return """
    <model name='{name}'>
      <static>1</static>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>{tile_size} {tile_size}</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>100</mu>
                <mu2>50</mu2>
              </ode>
              <torsional>
                <ode/>
              </torsional>
            </friction>
            <contact>
              <ode/>
            </contact>
            <bounce/>
          </surface>
          <max_contacts>10</max_contacts>
        </collision>
        <visual name='visual'>
          <cast_shadows>0</cast_shadows>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>{tile_size} {tile_size}</size>
            </plane>
          </geometry>
          <material>
            <script>
              <uri>model://.{road_name}/materials/scripts</uri>
              <uri>model://.{road_name}/materials/textures</uri>
              <name>{material}</name>
            </script>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
      <pose frame=''>{x} {y} 0 0 -0 0</pose>
    </model>
    """.format(
        x=x, y=y, tile_size=tile_size, name=name, road_name=road_name, material=material,
    )
