gazebo_simulation
=======================================================

The **gazebo_simulation** ROS package contains nodes that have a direct connection to Gazebo.
They receive information from Gazebo and are able to modify its behavior.

There are also the master.launch file inside the **launch**-folder which starts everything needed to run the simulation.

.. code-block::

  roslaunch gazebo_simulation master.launch

See :ref:`getting_started` for details the master.launch file.

Parameters used in nodes throughout this package are either defined in **param** or directly in the launch files used to start the node.

model_interface_node
------------------------
The ModelInterfaceNode class allows to set/get the poses and twists of models in gazebo.

.. code-block::

  roslaunch gazebo_simulation model_interface_node.launch

See :ref:`model_interface_node` for more details.

car_state_node
----------------
The car_state_node subscribes to Gazebo updates and publishes information
about the cars current state in shape of a CarStateMsg.
It can be launched by running

.. code-block::

  roslaunch gazebo_simulation car_state_node.launch

By default the parameter *rviz* is *true* and an additional visualization node is launched, displaying the car's frame and field of view in rviz.
See :ref:`car_state_node` for more details.

gazebo_rate_control_node
------------------------
The gazebo_rate_control_node can control Gazebo's maximum update rate to ensure that a specified topic publishes with a large enough rate.
It is primarily needed because Gazebo does not ensure a sensor update rate in Docker containers, which leads to low camera rates (e.g. *8Hz*).

This node is not started in the master.launch file by default. It can be manually started by passing the parameter *control_sim_rate:=true* or running:

.. code-block::

  roslaunch gazebo_simulation gazebo_rate_control_node.launch

See :ref:`gazebo_rate_control_node` for more details.



sensor_tof_node
------------------------
Gazebo does not provide distance sensors out of the box.
The SensorTofNode converts the output of a depth camera into a distance by publishing the distance to the closest object.

.. code-block::

  roslaunch gazebo_simulation sensor_tof_node.launch name:=NAME_OF_SENSOR topic:=OUTPUT_TOPIC

See :ref:`sensor_tof_node` for more details.
