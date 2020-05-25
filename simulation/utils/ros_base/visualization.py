#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for visualizing ROS messages in rviz."""

import rospy
import math

from visualization_msgs.msg import Marker

from simulation.utils.geometry.point import Point

from typing import List


def get_marker_for_points(
    points: List[Point],
    *,
    frame_id: str,
    type: int = 4,
    rgba: List[float] = [0, 0, 0, 1],
    id: int = 0,
    ns: str = None,
    duration: float = 1
) -> Marker:
    """ Rviz marker message from a list of points.

    Arguments:
        points (list(Point)): Points to visualize
        frame_id (str): Name of the points' coordinate frame (e.g. world, vehicle, simulation)
        type (int): Rviz marker type
        rgba (list(float)): Color of the marker
        id (int): Marker id
        ns (str): Rviz namespace
        duration (float): Marker will be shown for this long

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
    marker.scale.x = 0.02
    marker.pose.orientation.w = math.sqrt(1 - 0.000001 ** 2)
    marker.pose.orientation.z = 0.000001
    marker.id = id
    marker.type = type
    marker.points = [p.to_geometry_msg() for p in points]

    return marker
