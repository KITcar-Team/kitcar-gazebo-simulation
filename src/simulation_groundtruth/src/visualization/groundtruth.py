#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Node that publishes groundtruth information for rviz"""

# Import required Python code.
import rospy

# Received from groundtruth extractor
from simulation_groundtruth.msg import RoadSectionMsg
from geometry_msgs.msg import Point32

from visualization_msgs.msg import Marker #Used to publish to rviz


class VisualizationNode:
    """"""

    def __init__(self):

        rospy.init_node('visualization_node')

        #Read required parameters

        #Read optional parameters
        self.start_activated = rospy.get_param('~start_activated', True)

        if self.start_activated:
            self.start()

        rospy.spin()

        self.stop()

    def start(self):
        """Turn on publisher."""

        self.section_subscriber = rospy.Subscriber('/simulation/groundtruth/road_sections', RoadSectionMsg , callback=self.road_sections_cb, queue_size=100)

        self.rviz_left_line_publisher = rospy.Publisher('/simulation/visualization/left_lines', Marker, queue_size= 100)
        self.rviz_middle_line_publisher = rospy.Publisher('/simulation/visualization/middle_lines', Marker, queue_size= 100)
        self.rviz_right_line_publisher = rospy.Publisher('/simulation/visualization/right_lines', Marker, queue_size= 100)

            
    def stop(self):
        """Turn off publisher."""
        self.section_subscriber.unregister()
        pass


    def road_sections_cb(self, road_section_msg):
        """ Receive new road section message and publish to rviz """
        print(road_section_msg)

        #Convert straight road msg lines to rviz markers
        left_marker = self.get_line_marker(road_section_msg.left_line)
        middle_marker = self.get_line_marker(road_section_msg.middle_line)
        right_marker = self.get_line_marker(road_section_msg.right_line)

        self.rviz_left_line_publisher.publish(left_marker) 
        self.rviz_middle_line_publisher.publish(middle_marker)
        self.rviz_right_line_publisher.publish(right_marker)

        


    def get_line_marker(self, line):
        """ Turn a list of points into a marker for rviz
        @line [Point32]: list of geometry_msgs.Point32

        returns visualization_msgs.Marker
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.get_rostime()
        marker.ns = "simulation/visualization"
        marker.lifetime = rospy.Duration(secs = 10)
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1
        marker.scale.x = 0.02
        marker.scale.y = 1
        marker.scale.z = 1
        marker.type = 4 #List

        marker.points = line        
        
        return marker

