:orphan:

.. _car_state_node:

CarStateNode
=======================================================

The CarStateNode is a ROS node which subscribes to Gazebo updates and publishes information
about the cars current state in shape of a CarStateMsg.

.. autoclass:: car_state.node.CarStateNode
  :members:

There's also the option to visualize the current state in Rviz by launching the CarStateVisualizationNode:

.. autoclass:: car_state.visualization.CarStateVisualizationNode
  :members:
