#!/usr/bin/env python3
"""Utility functions for visualizing ROS messages in rviz."""

import math
from typing import List

import rospy
from visualization_msgs.msg import Marker

from simulation.utils.geometry.point import Point


def get_marker(
    frame_id: str,
    type: int = 4,
    rgba: List[float] = [0, 0, 0, 1],
    id: int = 0,
    ns: str = None,
    duration: float = 1,
    scale: float = 0.02,
):
    """Create a rviz marker message. The position or points can be added.

    Arguments:
        frame_id: Name of the points' coordinate frame
            (e.g. world, vehicle, simulation)
        type: Rviz marker type
        rgba: Color of the marker
        id: Marker id
        ns: Rviz namespace
        duration: Marker will be shown for this long

    Returns:
        Marker msg that rviz can display
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.get_rostime()
    if ns:
        marker.ns = ns
    marker.lifetime = rospy.Duration(secs=duration)
    marker.color.r = rgba[0]
    marker.color.g = rgba[1]
    marker.color.b = rgba[2]
    marker.color.a = rgba[3]
    marker.scale.x = scale
    marker.scale.y = scale
    marker.id = id
    marker.type = type
    marker.pose.orientation.w = math.sqrt(1 - 0.000001 ** 2)
    marker.pose.orientation.z = 0.000001
    return marker


def get_marker_for_points(
    points: List[Point],
    *,
    frame_id: str,
    type: int = 4,
    rgba: List[float] = [0, 0, 0, 1],
    id: int = 0,
    ns: str = None,
    duration: float = 1,
    scale: float = 0.02
) -> Marker:
    """Rviz marker message from a list of points.

    Arguments:
        points: Points to visualize
        frame_id: Name of the points' coordinate frame
            (e.g. world, vehicle, simulation)
        type: Rviz marker type
        rgba: Color of the marker
        id: Marker id
        ns: Rviz namespace
        duration: Marker will be shown for this long

    Returns:
        Marker msg that rviz can display
    """
    marker = get_marker(frame_id, type, rgba, id, ns, duration, scale)
    marker.points = [p.to_geometry_msg() for p in points]
    return marker
