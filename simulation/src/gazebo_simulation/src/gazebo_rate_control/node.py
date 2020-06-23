#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GazeboRateControlNode: Adjusts Gazebos update rate to guarantee sensor updates with desired frequency."""

import rospy

from gazebo_msgs.srv import (
    SetPhysicsPropertiesRequest,
    GetPhysicsPropertiesRequest,
    SetPhysicsProperties,
    GetPhysicsProperties,
)

import rostopic

from simulation.utils.ros_base.node_base import NodeBase


class GazeboRateControlNode(NodeBase):
    """Control gazebos update rate to meet desired sensor update rates.
    """

    def __init__(self):
        """Initialize the node"""

        super().__init__(name="gazebo_rate_control_node", log_level=rospy.INFO)

        self.run(function=self.update, rate=self.param.update_rate.control.rate)

    def start(self):
        self.set_physics = rospy.ServiceProxy(
            self.param.topics.set_physics, SetPhysicsProperties
        )
        self.get_physics = rospy.ServiceProxy(
            self.param.topics.get_physics, GetPhysicsProperties
        )

        self.rater = rostopic.ROSTopicHz(10)
        self.subscriber = rospy.Subscriber(
            self.param.topics.target,
            rospy.AnyMsg,
            self.rater.callback_hz,
            callback_args=self.param.topics.target,
        )
        super().start()

    def stop(self):
        self.subscriber.unregister()
        self.set_physics.close()
        self.get_physics.close()
        super().stop()

    def update(self):
        """Adjust Gazebos update rate to meet desired output frequency of the target topic.
        """

        frequency = self.rater.get_hz(self.param.topics.target)
        frequency = frequency[0] if frequency else None

        # Check if there's anything to do:
        if not frequency or (
            frequency > self.param.frequency.min and frequency < self.param.frequency.max
        ):
            return

        current_properties = self.get_physics(GetPhysicsPropertiesRequest())

        new_properties = SetPhysicsPropertiesRequest()
        new_properties.gravity = current_properties.gravity
        new_properties.ode_config = current_properties.ode_config
        new_properties.time_step = current_properties.time_step

        # Calculate new update rate
        if frequency < self.param.frequency.min:
            new_update_rate = (
                current_properties.max_update_rate
                - (current_properties.max_update_rate - self.param.update_rate.min)
                * self.param.update_rate.control.down
                * self.param.frequency.desired
                / frequency
            )
            new_update_rate = max(new_update_rate, self.param.update_rate.min)
        elif frequency > self.param.frequency.max:
            new_update_rate = (
                current_properties.max_update_rate
                + (self.param.update_rate.max - current_properties.max_update_rate)
                * self.param.update_rate.control.up
                * self.param.frequency.desired
                / frequency
            )
            new_update_rate = min(new_update_rate, self.param.update_rate.max)

        new_properties.max_update_rate = new_update_rate

        rospy.logdebug(f"NEW GAZEBO PHYSICS PROPERTIES: {new_properties}")
        rospy.logdebug(f"CURRENT CAMERA UPDATE RATE: {frequency} ")

        self.set_physics(new_properties)
