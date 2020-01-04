#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Contains CarStateNode: Publishes real time information about the car

Author: Konstantin Ditschuneit

Date: 07.11.2019

"""

import sys
import os
import rospy
import yaml  # parse config file

#### Calculations ####
import math
import numpy as np

from base.polygon import Polygon
from base.transform import Transform
from car_state.topics import Topic
import car_model.camera_calibration as calibration

#### Messages ####
from gazebo_msgs.msg import ModelStates
from gazebo_simulation.msg import CarStateMsg


class CarStateNode:
    """ Monitors and publishes the cars state

    @car_name:String

    @car_frame:shapely.geom.Polygon frame of car in vehicle coordinate system

    @view_cone:shapely.geom.Polygon view cone of car in vehicle coordinate system

    @subscriber:rospy.subscriber subscribes to gazebo/model_states

    @publisher:rospy.publisher(CarStateMsg) publishes real time information about the car

    """

    def __init__(self):
        """ initialize the node"""

        rospy.init_node('car_state_node')

        # Read required parameters
        # Number of points used to approximate the view cone
        self.view_cone_points = rospy.get_param('~cone_points')

        # Read optional parameters

        self.start_activated = rospy.get_param('~start_activated', True)
        self.car_name = rospy.get_param('~car_name', 'dr_drift')
        # Gazebo publishes with 1000 Hz, to reduce computations a max rate parameter can be set for this node
        self.max_rate = rospy.get_param('~max_rate', 120)

        car_config = rospy.get_param('~car_config', os.environ.get(
            'KITCAR_REPO_PATH') + '/kitcar-gazebo-simulation/models/model_db/dr_drift/dr_drift.yaml')
        self.read_configs(car_config)

        if self.start_activated:
            self.start()

        rospy.spin()

        self.stop()

    def start(self):
        """Start node."""

        self.last_update = 0  # Last time the car state was published

        self.publisher = rospy.Publisher(
            Topic.STATE, CarStateMsg, queue_size=1)

        # Start subscribing to changes in gazebo
        self.subscriber = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, callback=self.model_state_cb)

    def stop(self):
        """Turn off node."""
        self.subscriber.unregister()
        self.publisher.unregister()

    def read_configs(self, car_file):
        """ Read information about car specs from config file

        @car_file:string Path to config file 

        """

        with open(car_file, 'r') as stream:
            try:
                car_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print("Failed to open car_config: " + exc)
                return None

        ##### Car frame config #####
        chassis_size = car_config['chassis']['size']
        chassis_position = car_config['chassis']['pose']

        # get dimensions
        length = chassis_size[0]
        width = chassis_size[1]

        shift_x = chassis_position[0]
        shift_y = chassis_position[1]

        x1, y1 = length / 2 + shift_x, width / 2 + shift_y  # left front
        x2, y2 = x1, y1 - width  # right front
        x3, y3 = x2 - length, y2  # Right back
        x4, y4 = x3, y1  # Left back

        self.car_frame = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

        ##### Camera config #####

        # This parameter tells how far the camera can see
        view_distance = car_config['front_camera']['clip']['far']
        # Focal length of camera
        f = car_config['front_camera']['focal_length']
        # Number of pixels in horizontal direction
        w_pixels = car_config['front_camera']['output']['width']

        # Calculate field of view (opening angle of camera)
        fov = calibration.fov(f, w_pixels)

        view_cone_points = [(0, 0)]

        # Calculate a few points to approximate view frame
        # Add points on horizon of our camera (at view_distance away from vehicle) /approximates a view cone
        for alpha in np.linspace(-fov/2, fov/2, self.view_cone_points):

            x = math.cos(alpha)*view_distance
            y = math.sin(alpha)*view_distance
            view_cone_points.append((x, y))

        # Create geom.Polygon from points
        self.view_cone = Polygon(view_cone_points)

    def model_state_cb(self, model_states):
        """ Called when the model_states in gazebo are updated. Updates all car_state topics.

        @model_states:ModelStates 
        """

        # Ensure that this is called with a max rate of RATE
        current_time = rospy.Time.now().to_sec()
        if current_time - self.last_update < 1/self.max_rate:
            return
        else:
            self.last_update = current_time

        try:
            # Find information for current car in model_states and give to self.car_state
            idx = model_states.name.index(self.car_name)

            pose = model_states.pose[idx]
            twist = model_states.twist[idx]
        except:
            print(f"Failed to find {self.car_name} in gazebo models.")
            return

        # Transform
        frame = self.transformed_polygon_msg(pose, self.car_frame)
        view_cone = self.transformed_polygon_msg(pose, self.view_cone)

        # Create and publish CarStateMsg
        msg = CarStateMsg()
        msg.pose = pose
        msg.twist = twist
        msg.frame = frame
        msg.view_cone = view_cone

        if not rospy.is_shutdown():
            self.publisher.publish(msg)

    def transformed_polygon_msg(self, car_pose, frame):
        """
        Translate and rotate the frame to the cars pose

        @car_pose:geometry_msgs.Pose pose to transform to

        @frame:shapely.geom.Polygon

        return geometry_msgs.polygon transformed polygon as msg

        """

        tf = Transform(car_pose)

        # Create geometry_msgs/polygon with transformed points
        transformed_frame = tf*Polygon(frame.get_points())

        return transformed_frame.to_geometry_msg()
