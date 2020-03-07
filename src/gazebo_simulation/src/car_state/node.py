#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CarStateNode"""

import os
import rospy
import yaml  # parse config file
from functools import partial
import numpy as np

from ros_base.node_base import NodeBase
from geometry import Vector, Point, Polygon, Transform
import car_model.camera_calibration as calibration

import geometry_msgs.msg
from gazebo_simulation.msg import CarState as CarStateMsg
from gazebo_simulation.srv import GetModelPoseRequest, GetModelPose, GetModelTwistRequest, GetModelTwist

__copyright__ = "KITcar"


class CarStateNode(NodeBase):
    """ROS node which publishes information about the model in a CarState.

    ROS Parameters (by default they are defined in gazebo_simulation/param/...):
        * car_name (String): Name of the car_model in gazebo
        * max_rate (float): Max. update rate of the publisher
        * cone_points (int): Number of points used to approximate the vehicles view cone
        * car_config (str): Path to car config file.
        * topics:
            * gazebo_models(str): Topic on which gazebo publishes model states
            * car_state(str): Publisher topic of car_state

    Attributes:
        car_frame (shapely.geom.Polygon): Frame of car in vehicle coordinate system
        _vehicle_pose_proxy (rospy.ServiceProxy): ServiceProxy to get the cars pose from model_interface service
        get_vehicle_pose (Callable[[],None]: Returns current vehicle pose by calling the service proxy
        _vehicle_twist_proxy (rospy.ServiceProxy): ServiceProxy to get the cars twist from model_interface service
        get_vehicle_twist (Callable[[],None]: Returns current vehicle twist by calling the service proxy
        publisher (rospy.publisher): CarStateMsg) publishes real time information about the car
    """

    def __init__(self):
        """ initialize the node"""

        super(CarStateNode, self).__init__(name="car_state_node")

        self.read_car_config()

        # Start running node.
        self.run(function=self.state_update, rate=self.param.max_rate)

    def start(self):
        """Start node."""
        self.publisher = rospy.Publisher(self.param.topics.car_state, CarStateMsg, queue_size=1)

        rospy.wait_for_service(self.param.topics.model_interface.get.model_pose)
        rospy.wait_for_service(self.param.topics.model_interface.get.model_twist)

        # Use model interface service to get vehicles pose
        self._vehicle_pose_proxy = rospy.ServiceProxy(self.param.topics.model_interface.get.model_pose, GetModelPose)
        # Call proxy with car_name as input
        self.get_vehicle_pose = partial(self._vehicle_pose_proxy, GetModelPoseRequest(self.param.car_name))

        self._vehicle_twist_proxy = rospy.ServiceProxy(self.param.topics.model_interface.get.model_twist, GetModelTwist)
        # Call proxy with car_name as input
        self.get_vehicle_twist = partial(self._vehicle_twist_proxy, GetModelTwistRequest(self.param.car_name))

        super().start()

    def stop(self):
        """Turn off node."""
        super().stop()
        self._vehicle_pose_proxy.close()
        self._vehicle_twist_proxy.close()
        self.publisher.unregister()

    def read_car_config(self):
        """Process car parameters.

        """

        with open(os.environ.get("KITCAR_REPO_PATH") + self.param.car_config, "r") as stream:
            try:
                car_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print("Failed to open car_config: " + exc)
                return None

        """ Car frame config """
        chassis_size = Vector(car_config["chassis"]["size"])
        chassis_position = Point(car_config["chassis"]["pose"][0:3])

        # get dimensions
        x_span = Vector(0.5 * chassis_size.x, 0)  # Vector in x direction of length = width/2
        y_span = Vector(0, 0.5 * chassis_size.y)
        self.car_frame = Polygon(
            [
                chassis_position + x_span + y_span,  # Front right
                chassis_position - x_span + y_span,  # Front left
                chassis_position - x_span - y_span,  # Back left
                chassis_position + x_span - y_span,  # Back right
            ]
        )

        """ Camera config """

        # This parameter tells how far the camera can see
        view_distance: float = car_config["front_camera"]["clip"]["far"]
        # Focal length of camera
        f: float = car_config["front_camera"]["focal_length"]
        # Number of pixels in horizontal direction
        w_pixels: float = car_config["front_camera"]["output"]["width"]

        # Calculate field of view (opening angle of camera)
        fov: float = calibration.fov(f, w_pixels)

        if self.param.cone_points == 0:
            self.view_cone = None
            return

        # Calculate a few points to approximate view frame
        # Add points on horizon of our camera (at view_distance away from vehicle) /approximates a view cone

        # Create geom.Polygon from points
        self.view_cone = Polygon(
            [Point(0, 0)]
            + [Point(r=view_distance, phi=alpha) for alpha in np.linspace(-fov / 2, fov / 2, self.param.cone_points)]
        )

    def state_update(self):
        """Publish new CarState with updated information."""

        # Request current pose and twist from model_interface
        pose: geometry_msgs.msg.Pose = self.get_vehicle_pose().pose
        twist: geometry_msgs.msg.Twist = self.get_vehicle_twist().twist

        # Transform which is used to calculate frame and view cone
        tf = Transform(pose)

        # Create message
        msg = CarStateMsg()
        msg.pose = pose
        msg.twist = twist
        msg.frame = (tf * self.car_frame).to_geometry_msg()

        if self.view_cone:
            msg.view_cone = (tf * self.view_cone).to_geometry_msg()

        self.publisher.publish(msg)
