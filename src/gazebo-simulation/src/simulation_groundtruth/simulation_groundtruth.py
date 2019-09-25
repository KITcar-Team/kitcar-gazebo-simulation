#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example Python node to publish on a specific topic."""

import sys,os
sys.path.insert(1, os.path.join(os.environ['KITCAR_REPO_PATH'],'kitcar-gazebo-utils','road-generation'))

# Import required Python code.
import rospy
from std_msgs.msg import String
# Give ourselves the ability to run a dynamic reconfigure server.
from dynamic_reconfigure.server import Server as DynamicReconfigureServer

import commonroad.generator.primitive as primitive
import commonroad.generator.road_generation as road_generation
import commonroad.groundtruth.groundtruth_extractor as groundtruth_extractor
import numpy as np

from lxml import etree
import xml.dom.minidom

import pdb

import pkg_resources

SCHEMA = etree.XMLSchema(etree.parse(pkg_resources.resource_stream(
    "commonroad.generator", "template-schema.xsd")))


from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped, PolygonStamped


#shapely.geometry.Polygon(corridor.points)

class Config:
    def __init__(self):
        self.road_width = 0.4
        # TODO: move this to an enum in the generated pyxb schema
        self.turn_road_marking_width = 0.072


class Car_Config:
    def __init__(self):
        self.width = 0.2

class NodeBase(object):

    def loop(self):
        rospy.spin()
        return 0

    def start_module(self):
        raise NotImplementedError()

    def stop_module(self):
        raise NotImplementedError()

    def is_module_active(self):
        return self.module_active_

    def activate_if_desired(self):
        if self.is_module_active():
            self.start_module()



class SimulationGroundtruthNode(NodeBase):
    """Node example class."""

    def __init__(self):

        rospy.init_node('simulation_groundtruth')

        """Read in parameters."""
        # Get the private namespace parameters from the parameter server:
        # set from either command line or launch file.
        rate = rospy.get_param('~rate', 0.1)
        # Initialize enable variable so it can be used in dynamic reconfigure
        # callback upon startup.
        self.enable = True
        # Create a publisher for our custom message.
        self.pub = rospy.Publisher('example', String, queue_size=10)

        self.line_pub = rospy.Publisher('gazebo_simulation/groundtruth_lines', Marker,queue_size=200)

        self.corridor_pub = rospy.Publisher('gazebo_simulation/groundtruth_corridors', Marker,queue_size=100)

        self.simulation_groundtruth = SimulationGroundtruth('/home/ditschuk/kitcar/kitcar-gazebo-simulation/models/env_db/curves/road.xml')


        #get messages
        markers = self.simulation_groundtruth.get_line_messages()
        #pdb.set_trace()
        for marker in markers:
            #pdb.set_trace()
            self.line_pub.publish(marker)
        if self.enable:
            self.start()
        else:
            self.stop()

        # Create a timer to go to a callback at a specified interval.
        rospy.Timer(rospy.Duration(1.0 / rate), self.line_cb)
        rospy.Timer(rospy.Duration(0.3), self.corridor_cb)

        rospy.spin()

    def start(self):
        """Turn on publisher."""
        self.pub = rospy.Publisher('example',String, queue_size=10)

    def stop(self):
        """Turn off publisher."""
        self.pub.unregister()

    def line_cb(self, _event):
        """Call at a specified interval to publish message."""
        if not self.enable:
            return

        #get messages
        markers = self.simulation_groundtruth.get_line_messages()
        
        for marker in markers:
            rospy.sleep(0.001)
            #pdb.set_trace()
            self.line_pub.publish(marker)

    def corridor_cb(self, _event):

        corridors = self.simulation_groundtruth.get_corridor_messages()
        for corridor in corridors:
            self.line_pub.publish(corridor)


class SimulationGroundtruth:

    def __init__(self, road_path):
        self.road_path = road_path

        self.parse()


    def parse(self):

        primitives = self.parse_primitives(self.road_path)

        self.road_elements = groundtruth_extractor.extract_groundtruth(primitives,Config(),Car_Config())
        
        self.corridors = []
        self.lines = []
        for el in self.road_elements:
            self.corridors.append(el.corridor)
            self.lines += el.lines

        
    def parse_primitives(self, road_path):

        parser = etree.XMLParser(schema=SCHEMA)
        root = etree.parse(road_path, parser)

        primitives = road_generation.generate(root)

        return primitives

    def create_marker_from_line(self, line, id): 
        marker = Marker()
        marker.header.frame_id = "/world"
        marker.header.stamp = rospy.get_rostime()
        marker.ns = "simulation/line"
        marker.lifetime = rospy.Duration(secs = 10)
        marker.color.r = 1
        marker.color.g = 1
        marker.color.b = 1
        marker.color.a = 1
        marker.scale.x = 0.02
        marker.scale.y = 1
        marker.scale.z = 1
        marker.id = id

        marker.type = 4

        if line.marking == "dashed":
            marker.color.a = 0.3



        for point in line.points:
            state = PointStamped()
            state.point.x = point[0]
            state.point.y = point[1]
            state.point.z = 0
            #pdb.set_trace()

            marker.points.append(state.point)
        return marker
        
    def create_marker_from_corridor(self, corridor, id):
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
        marker.id = id

        marker.type = 5

        for point in corridor.points:
            state = PointStamped()
            state.point.x = point[0]
            state.point.y = point[1]
            state.point.z = 0
            #pdb.set_trace()

            marker.points.append(state.point)
        return marker

    def get_line_messages(self):
        markers = []
        for idx,line in enumerate(self.lines):
            markers.append(self.create_marker_from_line(line,id = idx))
        return markers

    def get_corridor_messages(self):
        markers = []
        for idx,corridor in enumerate(self.corridors):
            markers.append(self.create_marker_from_corridor(corridor,id = idx))
        return markers

    #def point_inside_corridor(self, point, corridor):



class CarStateHandler:
    """
    Used for listening to updates in Gazebo and publishing the state of the car.
    """
    
    def on_update(self):
        """ 
        When world is updated, the transformation of simulation coordinate system to vehicle,
        the cars position and velocity and 
        the polygon the car is
        """


