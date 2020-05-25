#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VehicleSimulationLinkNode"""

import rospy
from simulation.utils.ros_base.node_base import NodeBase
from simulation.utils.geometry import Transform, Vector

# Messages
from simulation_brain_link.msg import State as StateEstimationMsg
import geometry_msgs.msg
from gazebo_simulation.msg import (
    SetModelTwist as SetModelTwistMsg,
    SetModelPose as SetModelPoseMsg,
)
from tf2_msgs.msg import TFMessage

import std_msgs.msg

from pyquaternion import Quaternion

__copyright__ = "KITcar"


class VehicleSimulationLinkNode(NodeBase):
    """ROS node to translate state_estimation messages into movement of the vehicle in Gazebo
    and provide a simulation to world transformation.

    Whenever the pose of the vehicle is updated and a new message is received the updated twist is calculated
    and published.

    Attributes:
        set_model_twist_publisher (rospy.publisher): Publishes the new twist on the model plugins set_twist topic.
        get_model_pose_subscriber (rospy.subscriber): Receives the current pose of the model in Gazebo.
        state_estimation_subscriber (rospy.subscriber): Receives the state_estimation messages.
        latest_state_estimation (StateEstimationMsg): Latest received state estimation message.
        sub_tf (rospy.subscriber): Receive transformation updates
        pub_tf (rospy.publisher): Publish updates about the world to simulation transformation
        vehicle_world_tf (Transform): Transformation between the vehicle and world coordinate frame.
    """

    def __init__(self):
        self.vehicle_world_tf = Transform([0, 0], 0)
        self.speed = Vector(0, 0)
        self.yaw_rate = 0
        self.vehicle_simulation_rotation = Transform([0, 0], 0)

        super().__init__(
            name="vehicle_simulation_link_node"
        )  # Name can be overwritten in launch file

        self.run()

    def start(self):
        """Start node."""
        self.pub_tf = rospy.Publisher(
            "/tf", TFMessage, queue_size=100
        )  # See: https://github.com/ros/geometry2/blob/melodic-devel/tf2_ros/src/tf2_ros/transform_broadcaster.py
        self.sub_tf = rospy.Subscriber("/tf", TFMessage, self.receive_tf)

        self.set_model_twist_publisher = rospy.Publisher(
            self.param.topics.model_plugin.namespace
            + "/"
            + self.param.car_name
            + "/"
            + self.param.topics.model_plugin.set.twist,
            SetModelTwistMsg,
            queue_size=1,
        )
        self.set_model_pose_publisher = rospy.Publisher(
            self.param.topics.model_plugin.namespace
            + "/"
            + self.param.car_name
            + "/"
            + self.param.topics.model_plugin.set.pose,
            SetModelPoseMsg,
            queue_size=1,
        )
        self.state_estimation_subscriber = rospy.Subscriber(
            self.param.topics.state_estimation,
            StateEstimationMsg,
            self.state_estimation_cb,
            queue_size=1,
        )

        rospy.wait_for_message(
            self.param.topics.model_plugin.namespace
            + "/"
            + self.param.car_name
            + "/"
            + self.param.topics.model_plugin.get.pose,
            geometry_msgs.msg.Pose,
        )
        rospy.wait_for_message(self.param.topics.state_estimation, StateEstimationMsg)

        self.get_model_pose_subscriber = rospy.Subscriber(
            self.param.topics.model_plugin.namespace
            + "/"
            + self.param.car_name
            + "/"
            + self.param.topics.model_plugin.get.pose,
            geometry_msgs.msg.Pose,
            callback=self.receive_model_pose_cb,
            queue_size=1,
        )

        # Read initial position from parameters
        try:
            initial = self.param.initial_pose
            if len(initial) > 3:
                angle = initial[3]
                del initial[3]
            else:
                angle = 0
            pos = Vector(initial)
            self.initial_tf = Transform(pos, angle)
        except KeyError:
            self.initial_tf = None

        super().start()

    def stop(self):
        self.state_estimation_subscriber.unregister()
        self.set_model_twist_publisher.unregister()
        self.get_model_pose_subscriber.unregister()
        self.sub_tf.unregister()
        self.pub_tf.unregister()
        super().stop()

    def receive_model_pose_cb(self, msg: geometry_msgs.msg.Pose):
        """Receive new model pose."""
        rot = Quaternion(
            msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
        )
        if self.param.set_twist:
            self.update_twist(self.latest_state_estimation, rot)
        if self.param.set_pose:
            self.update_pose()

        self.update_simulation_world_tf(vehicle_simulation_tf=Transform(msg))

    def state_estimation_cb(self, msg: StateEstimationMsg):
        """Receive new state estimation."""
        self.latest_state_estimation = msg

    def update_twist(
        self, state_estimation: StateEstimationMsg, vehicle_simulation_rotation: Quaternion
    ):
        """Update the vehicle's twist by publishing an update on the model plugins set_twist topic.

        Args:
            state_estimation (StateEstimationMsg): latest state estimation message
            vehicle_simulation_rotation (Transform): latest update of the rotation between the simulation
                and vehicle coordinate frames
        """
        try:

            # The state_estimation message contains the speed in vehicle coordinates,
            # the vehicle world rotation converts the speed into simulation coordinates.
            speed_x, speed_y, _ = vehicle_simulation_rotation.rotate(
                (state_estimation.speed_x, state_estimation.speed_y, 0)
            )

            new_vals = []
            new_vals.append((SetModelTwistMsg.LINEAR_X, speed_x))
            new_vals.append((SetModelTwistMsg.LINEAR_Y, speed_y))
            new_vals.append(
                (SetModelTwistMsg.ANGULAR_Z, self.latest_state_estimation.yaw_rate)
            )

            keys, values = zip(*new_vals)
            msg = SetModelTwistMsg(keys, values)

            self.set_model_twist_publisher.publish(msg)

            rospy.logdebug(f"Updating the vehicle's twist with {new_vals}")

        except Exception as e:
            rospy.logerr(f"Error updating the vehicles twist {e}.")

    def update_pose(self):
        try:
            tf = self.vehicle_world_tf
            if self.initial_tf is None:
                tf = self.vehicle_world_tf
            else:
                tf = self.initial_tf * self.vehicle_world_tf

            new_vals = []
            new_vals.append((SetModelPoseMsg.POSITION_X, tf.x))
            new_vals.append((SetModelPoseMsg.POSITION_Y, tf.y))
            new_vals.append((SetModelPoseMsg.ORIENTATION_W, tf.rotation.w))
            new_vals.append((SetModelPoseMsg.ORIENTATION_X, tf.rotation.x))
            new_vals.append((SetModelPoseMsg.ORIENTATION_Y, tf.rotation.y))
            new_vals.append((SetModelPoseMsg.ORIENTATION_Z, tf.rotation.z))

            keys, values = zip(*new_vals)
            msg = SetModelPoseMsg(keys, values)

            self.set_model_pose_publisher.publish(msg)

            rospy.logdebug(f"Updating the vehicle's pose with {new_vals}")

        except Exception as e:
            rospy.logerr(f"Error updating the vehicles pose {e}.")

    def receive_tf(self, tf_msg: TFMessage):
        """Receive new message from the /tf topic and extract the world to vehicle transformation."""
        # Only select world to vehicle transformations
        def select_tf(tf):
            return (
                tf.header.frame_id == self.param.frame.world
                and tf.child_frame_id == self.param.frame.vehicle
            )

        # Select first tf (default: None)
        # Beware: Transformation goes from child to header frame !!
        vehicle_world: geometry_msgs.msg.TransformStamped = next(
            filter(select_tf, tf_msg.transforms), None
        )
        if vehicle_world:
            self.vehicle_world_tf = Transform(vehicle_world.transform)
            rospy.logdebug(
                f"Received vehicle to world transform {vehicle_world}"
                f"(Converted to: {self.vehicle_world_tf} {self.vehicle_world_tf.rotation})"
            )

    def update_simulation_world_tf(self, vehicle_simulation_tf: Transform):
        """Publish up to date simulation to world transformation to /tf.

        Args:
            vehicle_simulation_tf (Transform): Transformation between vehicle and simulation frames.
        """
        tf_stamped = geometry_msgs.msg.TransformStamped()

        tf_stamped.header = std_msgs.msg.Header()
        tf_stamped.header.stamp = rospy.Time.now()
        # Transform from world to simulation
        tf_stamped.header.frame_id = self.param.frame.simulation
        tf_stamped.child_frame_id = self.param.frame.world

        # Transformation from world to simulation == (world to vehicle -> vehicle to simulation)
        tf_stamped.transform = (
            vehicle_simulation_tf * self.vehicle_world_tf.inverse
        ).to_geometry_msg()

        self.pub_tf.publish(TFMessage([tf_stamped]))
