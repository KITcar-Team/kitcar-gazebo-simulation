:orphan:

.. _gazebo_rate_control_node:

GazeboRateControlNode
=======================================================

This node was created because Gazebo does not guarantee to update sensors at their desired frame rate.
A problem that seems to only arise in Docker containers is that the camera is only updated
with a frame rate of ~8-10 Hz making it impossible to use the simulation in Docker containers.

The GazeboRateControlNode is a ROS node attempting to control Gazebo's simulation speed, based on the update rate of a specified topic
(i.e. in this case the camera image topic */camera/image_raw*).

.. autoclass:: gazebo_rate_control.node.GazeboRateControlNode
  :members:
