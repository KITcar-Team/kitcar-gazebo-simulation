#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using the ModelInterfaceNode allows to easily control static and dynamic models inside the simulation
by providing a pose, a twist or a complete ModelState for a model inside the simulation.

There are a number of ROS services described below which enable the model interface's functionality.

However, in its current state the model interface comes with some issues.
It is good for getting model states, twists and poses.
But due to the fact that it is decoupled from Gazebo's update cycle, setting only the twist
(and not simultaneously the pose) of a model does not work very well.


"""

import rospy

# Messages
from gazebo_msgs.msg import ModelStates

import geometry_msgs.msg as geometry_msgs

import gazebo_msgs.srv as gazebo_srvs

from gazebo_simulation.srv import (
    SetModelPose,
    SetModelPoseRequest,
    SetModelPoseResponse,
    SetModelTwist,
    SetModelTwistRequest,
    SetModelTwistResponse,
    GetModelPose,
    GetModelPoseRequest,
    GetModelPoseResponse,
    GetModelTwist,
    GetModelTwistRequest,
    GetModelTwistResponse,
)

from ros_base.node_base import NodeBase

from cachetools.func import ttl_cache
from ratelimit import limits

__copyright__ = "KITcar"


class ModelInterfaceNode(NodeBase):
    """ROS node which listens to desired model states and passes them to gazebo.

    Attributes:
        model_state_subscriber (rospy.Subscriber): Subscribes to model states from gazebo
        set_gazebo_model_state (rospy.ServiceProxy): ROS service proxy which is used to set model states in gazebo.
        set_model_state_service (rospy.Service): ROS service which can be called to update
            the state of a model in Gazebo (It's just a helper for set_gazebo_model_state)
        set_model_pose_service (rospy.Service): ROS service which can be called to update the pose of a model
        set_model_twist_service (rospy.Service): ROS service which can be called to update the twist of a model
        get_model_pose_service (rospy.Service): ROS service which can be called to get the pose of a model
        get_model_twist_service (rospy.Service): ROS service which can be called to get the twist of a model

        _last_model_states (ModelStates): Attribute containing the last model states received from Gazebo.
            By subscribing to the states published by Gazebo, they can be used without actively asking Gazebo.
    """

    def __init__(self):

        super().__init__(name="model_interface_node")  # Name can be overwritten in launch file

        self.run()

    def start(self):
        self.model_state_subscriber = rospy.Subscriber(
            self.param.topics.gazebo_models,
            ModelStates,
            callback=limits(calls=1, period=1 / self.param.max_rate, raise_on_limit=False)(self.model_state_cb),
            queue_size=1,
        )

        # Don't advertise the services before the first message from gazebo is received
        rospy.wait_for_message(self.param.topics.gazebo_models, ModelStates)

        self.set_gazebo_model_state = rospy.ServiceProxy(self.param.topics.gazebo_set_state, gazebo_srvs.SetModelState)

        self.set_model_state_service = rospy.Service(
            self.param.topics.set.model_state, service_class=gazebo_srvs.SetModelState, handler=self.update_model_state,
        )
        self.set_model_pose_service = rospy.Service(
            self.param.topics.set.model_pose, service_class=SetModelPose, handler=self.update_model_pose
        )
        self.set_model_twist_service = rospy.Service(
            self.param.topics.set.model_twist, service_class=SetModelTwist, handler=self.update_model_twist
        )

        self.get_model_pose_service = rospy.Service(
            self.param.topics.get.model_pose, service_class=GetModelPose, handler=self.get_model_pose
        )
        self.get_model_twist_service = rospy.Service(
            self.param.topics.get.model_twist, service_class=GetModelTwist, handler=self.get_model_twist
        )

        super().start()

    def stop(self):
        super().stop()

        self.model_state_subscriber.unregister()
        self.set_gazebo_model_state.close()

        self.set_model_state_service.shutdown()
        self.set_model_pose_service.shutdown()
        self.set_model_twist_service.shutdown()

        self.get_model_pose_service.shutdown()
        self.get_model_twist_service.shutdown()

    def model_state_cb(self, model_states: ModelStates):
        """ Called when the model_states in gazebo are updated. Updates all car_state topics.

        Arguments:
            model_states (ModelStates): List of gazebo models
        """
        self._last_model_states = model_states

    # Cache model names to reduce overhead
    @ttl_cache(ttl=5)
    def _model_name_idx(self, name: str) -> int:
        """Get the index of a model in the gazebo ModelStates message.
        The index changes only when adding or removing a model in Gazebo.
        This function allows to cache the index."""
        return self._last_model_states.name.index(name)

    def last_pose_of_model(self, *, name: str) -> geometry_msgs.Pose:
        """Last received pose of a model in Gazebo.

        Arguments:
            name (str): name of the model

        Returns:
            The model's pose.

        Raises:
            KeyError if model is not in Gazebo's model list.
        """
        return self._last_model_states.pose[self._model_name_idx(name)]

    def last_twist_of_model(self, *, name: str) -> geometry_msgs.Twist:
        """Last received twist of a model in Gazebo.

        Arguments:
            name (str): name of the model

        Returns:
            The model's twist.

        Raises:
            KeyError if model is not in Gazebo's model list.
        """
        return self._last_model_states.twist[self._model_name_idx(name)]

    def update_model_state(self, request: gazebo_srvs.SetModelStateRequest) -> gazebo_srvs.SetModelStateResponse:
        """Update the state of a model in gazebo.

        The request is passed to gazebo without changing anything. Basically just a helper function.

        Arguments:
            request (gazebo_srvs.SetModelStateRequest): ROS service request
        """
        rospy.logdebug("Requested update of {request.model_name}'s state with {request}")
        self.set_gazebo_model_state(request)

        return gazebo_srvs.SetModelStateResponse()

    def update_model_pose(self, request: SetModelPoseRequest) -> SetModelPoseResponse:
        """Update the pose of a model in gazebo, while keeping all other properties unchanged.

        Arguments:
            request (SetModelPoseRequest): ROS service request
        """
        rospy.logdebug("Requested update of {request.model_name}'s pose with {request.pose}")

        # Read last state
        state_request = gazebo_srvs.SetModelStateRequest()
        state_request.model_state.pose = request.pose
        state_request.model_state.twist = self.last_twist_of_model(
            name=request.model_name
        )  # Append the last known twist
        state_request.model_state.model_name = request.model_name
        state_request.model_state.reference_frame = ""

        # Update state in gazebo
        self.update_model_state(request=state_request)

        return SetModelPoseResponse()

    def update_model_twist(self, request: SetModelTwistRequest) -> SetModelTwistRequest:
        """Update the twist of a model in gazebo, while keeping all other properties unchanged.

        Arguments:
            request (SetModelTwistRequest): ROS service request
        """
        rospy.logdebug("Requested update of {request.model_name}'s twist with {request.twist}")

        # Read last state
        state_request = gazebo_srvs.SetModelStateRequest()
        state_request.model_state.pose = self.last_pose_of_model(name=request.model_name)
        state_request.model_state.twist = request.twist
        state_request.model_state.model_name = request.model_name
        state_request.model_state.reference_frame = ""

        # Update state in gazebo
        self.update_model_state(request=state_request)

        return SetModelTwistResponse()

    def get_model_pose(self, request: GetModelPoseRequest) -> GetModelPoseResponse:
        """Get the pose of a model in gazebo.

        Arguments:
            request (GetModelPoseRequest): ROS service request

        Returns:
            Last received pose of the model.
        """
        rospy.logdebug("Requested {request.model_name}'s pose.")

        pose = self.last_pose_of_model(name=request.model_name)

        return GetModelPoseResponse(pose)

    def get_model_twist(self, request: GetModelTwistRequest) -> GetModelTwistResponse:
        """Get the twist of a model in gazebo.

        Arguments:
            request (GetModelTwistRequest): ROS service request

        Returns:
            Last received twist of the model.
        """
        rospy.logdebug("Requested {request.model_name}'s twist.")

        twist = self.last_twist_of_model(name=request.model_name)

        return GetModelTwistResponse(twist)
