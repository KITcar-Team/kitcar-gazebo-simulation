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
import pyquaternion

import shapely.geometry as geom
import shapely.affinity as affinity
import shapely.ops as ops

from gazebo_msgs.srv import (
    SetPhysicsPropertiesRequest,
    GetPhysicsPropertiesRequest,
    SetPhysicsProperties,
    GetPhysicsProperties,
)

import rostopic


class GazeboTimerNode:
    """ Monitors and publishes the cars state

    """

    def __init__(self):
        """ initialize the node"""

        rospy.init_node("gazebo_timer_node")

        # Read required parameters
        # Number of points used to approximate the view cone
        self.maximum_frequency = rospy.get_param("~max_rate", 55)
        self.minimum_frequency = rospy.get_param("~min_rate", 45)

        self.desired_frequency = \
            0.5 * (self.minimum_frequency + self.maximum_frequency)

        self.min_update_rate = rospy.get_param("~min_update_rate", 10)
        self.max_update_rate = rospy.get_param("~max_update_rate", 1500)

        self.p_down_factor = rospy.get_param("~p_down", 0.05)
        self.p_up_factor = rospy.get_param("~p_up", 0.001)

        # Read optional parameters

        self.start_activated = rospy.get_param("~start_activated", True)

        if self.start_activated:
            self.start()

            while not rospy.is_shutdown():

                rospy.sleep(0.03)

                self.update()

    def start(self):
        """Start node."""

        self.set_physics = rospy.ServiceProxy(
            "/gazebo/set_physics_properties", SetPhysicsProperties
        )
        self.get_physics = rospy.ServiceProxy(
            "/gazebo/get_physics_properties", GetPhysicsProperties
        )

        self.rater = rostopic.ROSTopicHz(10)
        self.subscriber = rospy.Subscriber(
            "/camera/image_raw",
            rospy.AnyMsg,
            self.rater.callback_hz,
            callback_args="/camera/image_raw",
        )

    def update(self):

        freq = self.rater.get_hz("/camera/image_raw")

        if not freq or (
            freq[0] > self.minimum_frequency and
                freq[0] < self.maximum_frequency
        ):
            return

        current_properties = self.get_physics(GetPhysicsPropertiesRequest())

        # Calculate new update rate

        new_properties = SetPhysicsPropertiesRequest()

        new_properties.gravity = current_properties.gravity
        new_properties.ode_config = current_properties.ode_config
        new_properties.time_step = current_properties.time_step

        if freq[0] < self.minimum_frequency:
            new_properties.max_update_rate = (
                current_properties.max_update_rate -
                (current_properties.max_update_rate - self.min_update_rate) *
                self.p_down_factor * self.desired_frequency / freq[0]
            )
        elif freq[0] > self.maximum_frequency:
            new_properties.max_update_rate = (
                current_properties.max_update_rate +
                (self.max_update_rate - current_properties.max_update_rate) *
                self.p_up_factor * self.desired_frequency / freq[0]
            )
        else:
            return

        if new_properties.max_update_rate < self.min_update_rate:
            new_properties.max_update_rate = self.min_update_rate

        if new_properties.max_update_rate > self.max_update_rate:
            new_properties.max_update_rate = self.max_update_rate

        rospy.logdebug(f"NEW GAZEBO PHYSICS PROPERTIES: {new_properties}")
        rospy.logdebug(f"CURRENT CAMERA UPDATE RATE: {freq} ")

        self.set_physics(new_properties)
