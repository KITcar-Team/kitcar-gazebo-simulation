#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example Python node to publish on a specific topic."""

import sys,os
sys.path.insert(1, os.path.join(os.environ['KITCAR_REPO_PATH'],'kitcar-gazebo-utils','road-generation'))


# Import required Python code.
import rospy
# Give ourselves the ability to run a dynamic reconfigure server.
from dynamic_reconfigure.server import Server as DynamicReconfigureServer

import commonroad.generator.primitive as primitive
from commonroad.generator.primitive import RoadElement
import commonroad.generator.road_generation as road_generation

import numpy as np

from lxml import etree
import xml.dom.minidom

import pkg_resources

SCHEMA = etree.XMLSchema(etree.parse(pkg_resources.resource_stream(
    "commonroad.generator", "template-schema.xsd")))


from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped, PolygonStamped
from std_msgs.msg import String

#shapely.geometry.Polygon(corridor.points)

class Config:
    def __init__(self):
        self.road_width = 0.4
        # TODO: move this to an enum in the generated pyxb schema
        self.turn_road_marking_width = 0.072


class SimulationGroundtruthNode:
    """"""

    def __init__(self):

        rospy.init_node('simulation_groundtruth_node')

        #Read required parameters
        road = rospy.get_param('~road')
        self.car_name = rospy.get_param('~car_name')

        #Read optional parameters
        self.topic_env = rospy.get_param('~topic_environment','/simulation/groundtruth/')
        self.start_activated = rospy.get_param('~start_activated', True)
        self.rate = rospy.get_param('~rate', 1)

        road_file = os.path.join(os.environ.get('KITCAR_REPO_PATH'),'kitcar-gazebo-simulation','models','env_db',rospy.get_param('~road','curves'),'road.xml')


        self.groundtruth = SimulationGroundtruth(road_file)
        self.groundtruth_updater = SimulationGroundtruthUpdater(self.topic_env + 'lines', self.topic_env+'corridor')

        if self.start_activated:
            self.start()

        rospy.spin()

        self.stop()

    def start(self):
        """Turn on publisher."""
       
        # Create a timer to go to a callback at a specified interval.
        rospy.Timer(rospy.Duration(1.0 / self.rate), self.line_cb)
     
    def stop(self):
        """Turn off publisher."""
        pass

    def line_cb(self, _event):
        """Called to update groundtruth lines."""
        
        lines = self.groundtruth.lines

        self.groundtruth_updater.publish_lines(lines)

        #test

class SimulationGroundtruth:
    """ Extracts and stores information about the track used in simulation 

    @road_elements: [commonroad.groundtruth.RoadElement] containing line markings corresponding to road 
    
    @corridors: [commonroad.groundtruth.corridor] which holds points of a polygon. The polygons in total describe the drivable path the vehicle can take

    """

    def __init__(self, road_file):
        self.load(road_file)

    def load(self, road_file):
        """ Load and parse file at @road_path to exract groundtruth information. """

        primitives = self.parse_primitives(road_file)

        self.road_elements = [RoadElement(lines= primitive.export_lines(Config()), corridor= primitive.export_corridor(Config())) for primitive in primitives]
        
        self.corridors = []
        self.lines = []

        for el in self.road_elements: #Each road element contains multiple lines, but only a single drivable corridor 
            self.corridors.append(el.corridor)
            self.lines += el.lines 

        
    def parse_primitives(self, road_file):
        """ Load and parse commonroad.generator.primitive(s) from file @road_path. 

        Return: List of primitives, that make up the road.
        
        """
        
        parsed_road = etree.parse(road_file, etree.XMLParser(schema=SCHEMA))

        primitives = road_generation.generate(parsed_road)

        return primitives


class SimulationGroundtruthUpdater:

    def __init__(self, line_topic, corridor_topic):
        self.line_publisher = rospy.Publisher(line_topic, Marker, queue_size = 200)
        self.corridor_publisher = rospy.Publisher(corridor_topic, Marker, queue_size = 200)

    def publish_lines(self,lines):
        """ Publish the lines as visualization_msgs.msg.Marker to easily display in rviz. """
        markers = self.get_line_markers(lines)

        for marker in markers:
            rospy.sleep(0.001) # To prevent flooding of topic
            self.line_publisher.publish(marker)

    def publish_corridors(self,corridors):
        """ Publish the corridors as visualization_msgs.msg.Marker to easily display in rviz. """
        markers = self.get_corridor_markers(corridors)

        for marker in markers:
            rospy.sleep(0.001) # To prevent flooding of topic
            self.corridor_publisher.publish(marker)

    def get_line_markers(self, lines):
        markers = []
        for idx,line in enumerate(lines):

            marker = Marker()
            marker.header.frame_id = "simulation"
            marker.header.stamp = rospy.get_rostime()
            marker.ns = "simulation/line"
            marker.lifetime = rospy.Duration(secs = 10)
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 1
            marker.scale.x = 0.02
            marker.scale.y = 1
            marker.scale.z = 1
            marker.id = idx

            marker.type = 4 #List

            if line.marking == "dashed":
                marker.color.a = 0.3

            for point in line.coords:
                state = PointStamped()
                state.point.x = point[0]
                state.point.y = point[1]
                state.point.z = 0

                marker.points.append(state.point)

            markers.append(marker)
        
        return markers

    def get_corridor_markers(self, corridors):
        markers = []
        
        for idx,corridor in enumerate(corridors):

            marker = Marker()
            marker.header.frame_id = "/world"
            marker.header.stamp = rospy.get_rostime()
            marker.ns = "simulation/corridor"
            marker.lifetime = rospy.Duration(secs = 10)
            marker.color.r = 1
            marker.color.g = 1
            marker.color.b = 1
            marker.color.a = 0.8
            marker.scale.x = 0.02
            marker.scale.y = 1
            marker.scale.z = 1
            marker.id = idx

            marker.type = 5 #Polygon 

            for point in corridor.exterior.coords:
                state = PointStamped()
                state.point.x = point[0]
                state.point.y = point[1]
                state.point.z = 0

                marker.points.append(state.point)

            markers.append(marker)
        return markers

    #def point_inside_corridor(self, point, corridor):

