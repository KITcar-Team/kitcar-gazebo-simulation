:orphan:

.. _sensor_tof_node:

SensorTofNode
=======================================================

Gazebo does not provide distance sensors out of the box. As a workaround, the simulated `Dr. Drift` is equipped with depth cameras.
The depth camera sensor data is then converted into a distance by extracting the closest point inside the depth cameras point cloud.

This is done separately for each time of flight sensor through an instance of the SensorTofNode:

.. autoclass:: sensor_tof.node.SensorTofNode
  :members:
  :exclude-members: start, stop
