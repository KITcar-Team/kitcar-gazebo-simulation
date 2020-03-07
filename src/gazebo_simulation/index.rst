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
