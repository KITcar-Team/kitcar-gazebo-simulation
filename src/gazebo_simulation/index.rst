gazebo_simulation
=======================================================

The **gazebo_simulation** ROS package contains nodes that have a direct connection to Gazebo.
They receive information from Gazebo and are able to modify its behavior.

There are also the master.launch file inside the **launch**-folder which starts everything needed to run the simulation.

.. code-block::

  roslaunch gazebo_simulation master.launch

See :ref:`getting_started` for details the master.launch file.

Parameters used in nodes throughout this package are either defined in **param** or directly in the launch files used to start the node.
