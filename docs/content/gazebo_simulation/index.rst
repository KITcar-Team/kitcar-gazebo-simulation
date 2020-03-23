gazebo_simulation
=======================================================

This package contains ROS nodes used as abstraction layers to provide easy access to Gazebo. Also, there are multiple launch files which start all necessary components to launch the simulation.


Launch Files
--------------

The main launch file for starting `kitcar-gazebo-simulation` related nodes is the **master.launch**. By default it starts all necessary components needed to run the simulation.

.. code-block::

  roslaunch gazebo_simulation master.launch

See :ref:`getting_started` or checkout the source file for available parameters.

Nodes
----------------

The car_state_node subscribes to Gazebo updates and publishes information
about the cars current state in shape of a CarStateMsg.
It can be launched by running

.. code-block::

  roslaunch gazebo_simulation car_state_node.launch

By default the parameter *rviz* is *true* and an additional visualization node is launched, displaying the car's frame and field of view in rviz.
See :mod:`simulation.src.gazebo_simulation.src.car_state` for implementation details.


The gazebo_rate_control_node can control Gazebo's maximum update rate to ensure that a specified topic publishes with a large enough rate.
It is primarily needed because Gazebo does not ensure a sensor update rate in Docker containers, which leads to low camera rates (e.g. *8Hz*).

This node is not started in the master.launch file by default. It can be manually started by passing the parameter *control_sim_rate:=true* or running:

.. code-block::

  roslaunch gazebo_simulation gazebo_rate_control_node.launch

See :mod:`simulation.src.gazebo_simulation.src.gazebo_rate_control.node` for implementation details.
