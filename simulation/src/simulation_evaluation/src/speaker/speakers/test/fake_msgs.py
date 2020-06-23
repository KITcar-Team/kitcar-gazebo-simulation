from typing import List
from simulation.utils.geometry import Line, Vector, Polygon, Point
import math
import itertools
import random

import simulation_groundtruth.msg as groundtruth_msgs
import simulation_groundtruth.srv as groundtruth_srvs

from simulation.utils.road.sections.line_tuple import LineTuple


def create_points(
    start_point=Point(0, 0),
    offset_right=1,
    point_dist=1,
    section_count=10,
    point_count=100,
    direction=0,
    deviation=math.pi / 8,
) -> List[LineTuple]:
    """Create a line of points and split it up into multiple sections."""

    # Points that are biased in one direction
    points = [start_point] + list(
        Vector(r=point_dist, phi=direction + (random.random() - 0.5) * 2 * deviation)
        for _ in range(point_count)
    )
    # Prefix sum over the points to get the middle line points
    middle_points = list(itertools.accumulate(points))

    # Divide the middle points into multiple sections
    def divide_chunks():
        for i in range(0, point_count, int(point_count / section_count)):
            yield middle_points[
                i : i + int(point_count / section_count) + 1  # noqa: 203
            ]  # Also include first point of next section

    middle_lines = (Line(ps) for ps in divide_chunks())

    return list(
        LineTuple(
            middle_line.parallel_offset(offset_right, side="left"),
            middle_line,
            middle_line.parallel_offset(offset_right, side="right"),
        )
        for middle_line in middle_lines
    )


def section_msgs(section_count: int = 4, section_types: List[int] = None):
    section_msgs = groundtruth_srvs.SectionSrvResponse()

    for i in range(0, section_count):
        sec = groundtruth_msgs.Section()
        sec.id = i
        if section_types:
            sec.type = section_types[i]
        else:
            sec.type = 5  # Parking

        section_msgs.sections.append(sec)

    return section_msgs


def section_srv(section_count: int = 4, section_types: List[int] = None):
    def _cal(req: groundtruth_srvs.SectionSrvRequest):
        return section_msgs(section_count, section_types)

    return _cal


def lane_msgs(
    lines: List[LineTuple] = [], id: int = 0
) -> groundtruth_srvs.LaneSrvResponse:  # Don't care about the default value list
    """Imitate the response of the groundtruth lane service by returning what's in lines at idx id."""
    lane_msg = groundtruth_msgs.Lane()

    lane_msg.left_line = lines[id].left.to_geometry_msgs()
    lane_msg.middle_line = lines[id].middle.to_geometry_msgs()
    lane_msg.right_line = lines[id].right.to_geometry_msgs()

    lane_srv_response = groundtruth_srvs.LaneSrvResponse()
    lane_srv_response.lane_msg = lane_msg

    return lane_srv_response


def lane_srv(lines=None):
    if not lines:
        lines = []

    def _cal(req: groundtruth_srvs.LaneSrvRequest):
        return lane_msgs(lines, req.id)

    return _cal


def obstacle_msgs(
    obstacles: List[Polygon], id: int
) -> groundtruth_srvs.LabeledPolygonSrvResponse:
    response = groundtruth_srvs.LabeledPolygonSrvResponse()
    response.polygons = list(
        groundtruth_msgs.LabeledPolygon(
            obtacle.to_geometry_msg(), groundtruth_msgs.LabeledPolygon.OBSTACLE
        )
        for obtacle in obstacles
    )

    return response


def obstacle_srv(obstacles=None):
    if not obstacles:
        obstacles = []

    def _cal(req: groundtruth_srvs.LabeledPolygonSrvResponse):
        return obstacle_msgs(obstacles, req.id)

    return _cal


def intersection_msg(
    south: groundtruth_msgs.Lane = None,
    west: groundtruth_msgs.Lane = None,
    east: groundtruth_msgs.Lane = None,
    north: groundtruth_msgs.Lane = None,
    rule: int = 0,
    turn: int = 0,
    id: int = 0,
) -> groundtruth_srvs.IntersectionSrvResponse:
    res = groundtruth_srvs.IntersectionSrvResponse()
    res.rule = rule

    if south:
        res.south = south
    if west:
        res.west = west
    if east:
        res.east = east
    if north:
        res.north = north
    res.rule = rule
    res.turn = turn

    return res


def intersection_srv(
    south: groundtruth_msgs.Lane = None,
    west: groundtruth_msgs.Lane = None,
    east: groundtruth_msgs.Lane = None,
    north: groundtruth_msgs.Lane = None,
    rule: int = 0,
    turn: int = 0,
):
    def _cal(req: groundtruth_srvs.IntersectionSrvRequest):
        return intersection_msg(south, west, east, north, rule, turn, req.id)

    return _cal


def parking_msgs(
    right_spots: List[Polygon],
    left_spots: List[Polygon],
    right_border: Line,
    left_border: Line,
    id: int,
) -> groundtruth_srvs.ParkingSrvResponse:
    response = groundtruth_srvs.ParkingSrvResponse()

    right_msg = groundtruth_msgs.Parking()
    right_msg.borders = [groundtruth_msgs.Line(right_border.to_geometry_msgs())]
    right_msg.spots = list(
        groundtruth_msgs.LabeledPolygon(
            spot.to_geometry_msg(), groundtruth_msgs.Parking.SPOT_FREE
        )
        for spot in right_spots
    )
    response.right_msg = right_msg

    left_msg = groundtruth_msgs.Parking()
    left_msg.borders = [groundtruth_msgs.Line(left_border.to_geometry_msgs())]
    left_msg.spots = list(
        groundtruth_msgs.LabeledPolygon(
            spot.to_geometry_msg(), groundtruth_msgs.Parking.SPOT_FREE
        )
        for spot in left_spots
    )
    response.left_msg = left_msg

    return response


def parking_srv(
    right_spots: List[Polygon],
    left_spots: List[Polygon],
    right_border: Line,
    left_border: Line,
):
    def _cal(req: groundtruth_srvs.ParkingSrvRequest):
        return parking_msgs(right_spots, left_spots, right_border, left_border, req.id)

    return _cal
