#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Contains CarStateVisualizationNode: Visualizes real time information about the car

Author: Konstantin Ditschuneit

Date: 14.11.2019

"""
import rospy
import math

#### Messages ####
from visualization_msgs.msg import Marker
from gazebo_simulation.msg import CarStateMsg

#### Topics ####
from car_state.topics import Topic


class CarStateVisualizationNode:

    def __init__(self):
        """ initialize the node"""

        rospy.init_node('car_state_visualization_node')

        # Read optional parameters
        self.start_activated = rospy.get_param('~start_activated', True)

        if self.start_activated:
            self.start()

        rospy.spin()

        self.stop()

    def start(self):
        """ Start visualization. """
        self.frame_publisher = rospy.Publisher(
            Topic.VISUALIZATION_CONE, Marker, queue_size=1)
        self.view_cone_publisher = rospy.Publisher(
            Topic.VISUALIZATION_FRAME, Marker, queue_size=1)
        self.state_subscriber = rospy.Subscriber(
            Topic.STATE, CarStateMsg, callback=self.state_cb)

    def stop(self):
        """ Stop visualization. """
        self.state_subscriber.unregister()
        self.frame_publisher.unregister()
        self.view_cone_publisher.unregister()

    def state_cb(self, msg):
        """ Called when car state is published

        @msg:CarStateMsg 

        """

        frame = msg.frame
        self.publish_polygon(frame, self.frame_publisher, [0, 0, 1, 0.7])

        cone = msg.view_cone
        self.publish_polygon(cone, self.view_cone_publisher, id=1)

    def publish_polygon(self, polygon, publisher, rgba=[0, 0, 0, 1], id=0):
        """
        Publish @polygon:geometry_msgs.Polygon at publisher:rospy.Publisher in color @rgba:[float]
        """
        marker = Marker()
        marker.header.frame_id = "simulation"
        marker.header.stamp = rospy.get_rostime()
        marker.ns = "car_state"
        marker.lifetime = rospy.Duration(secs=1)
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.scale.x = 0.02
        marker.pose.orientation.w = math.sqrt(1-0.000001**2)
        marker.pose.orientation.z = 0.000001
        marker.id = id

        marker.type = 4  # Polygon
        marker.points = polygon.points

        publisher.publish(marker)
