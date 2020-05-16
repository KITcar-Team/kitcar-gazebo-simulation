.. _gazebo_simulation:

gazebo_simulation
=======================================================

ROS package to better communicate with Gazebo.

.. program-output:: cd $KITCAR_REPO_PATH/kitcar-gazebo-simulation && tree -a -I __pycache__ --dirsfirst simulation/src/gazebo_simulation
   :shell:


.. toctree::
   :maxdepth: 1
   :caption: Packages and Modules

   ../_source_files/simulation.src.gazebo_simulation.src.car_state
   ../_source_files/simulation.src.gazebo_simulation.src.gazebo_rate_control


master.launch
-------------

The :ref:`master.launch master_launch` file includes all necessary components to run the \
simulation locally.
Refer to the actual source code for more details on available parameters.

.. admonition:: Launch

   Start the complete simulation with

   .. prompt:: bash

      roslaunch gazebo_simulation master.launch

.. seealso::

   :ref:`getting_started` for more details.

CarStateNode
--------------

The :py:class:`simulation.src.gazebo_simulation.src.car_state.node.CarStateNode` \
subscribes to Gazebo updates and publishes information about the cars current state
(*position + speed + ...*) in shape of a :ref:`car_state_msg`.

.. admonition:: Launch

   .. prompt:: bash

      roslaunch gazebo_simulation car_state_node.launch


By default the parameter *rviz* is *true* and \
:py:class:`simulation.src.gazebo_simulation.src.car_state.visualization.CarStateVisualizationNode` \
is launched as well.
Enabling to display the car's frame and field of view in RVIZ.

GazeboRateControlNode
------------------------

The :py:class:`simulation.src.gazebo_simulation.src.gazebo_rate_control.node.GazeboRateControlNode` \
can control Gazebo's maximum update rate to ensure that a specified topic \
publishes with a large enough rate.
It is primarily necessary because Gazebo does not ensure \
a sensor update rate in Docker containers.

This node is not started in the master.launch file by default.
It can be manually started by passing the parameter *control_sim_rate:=true* or running:

.. admonition:: Launch

   .. prompt:: bash

      roslaunch gazebo_simulation gazebo_rate_control_node.launch


ModelPluginLinkNode
----------------------

The :ref:`model_plugin_link_node` is a \
`Gazebo model plugin <http://gazebosim.org/tutorials?tut=plugins_model&cat=write_plugin>`_
that allows to interact with models in Gazebo easily.

It can be attached to Gazebo models by adding

.. code-block:: xml

   <plugin filename="libmodel_plugin_link.so" name="model_plugin_link"/>

to the *model.sdf*.

When Gazebo loads a model with the *model_plugin_link*, a new instance of the model plugin \
link node is created.
The model plugin link node creates two publisher's:

- */simulation/gazebo/model/MODEL_NAME/pose* \
  (`geometry_msgs/Pose.msg <http://docs.ros.org/melodic/api/geometry_msgs/html/msg/Pose.html>`_): \
  Publish the model's pose.
- */simulation/gazebo/model/MODEL_NAME/twist* \
  (`geometry_msgs/Twist.msg <http://docs.ros.org/melodic/api/geometry_msgs/html/msg/Twist.html>`_): \
  Publish the model's twist.

And two subscribers:

- */simulation/gazebo/model/MODEL_NAME/set_pose* (:ref:`set_model_pose_msg`): \
  Receive new model pose.
- */simulation/gazebo/model/MODEL_NAME/set_twist* (:ref:`set_model_twist_msg`): \
  Receive new model twist.

.. graphviz::
   :align: center
   :caption: Schema of the Model Plugin Link

   digraph ModelPluginLink {

     node [style=dotted, shape=box]; model [label="Gazebo model"];
     node [style=solid, shape=ellipse]; model_plugin_link_node [label="model_plugin_link"];
     node [shape=box]; pose_topic [label="pose"]; twist_topic [label="twist"];set_pose_topic [label="set_pose"]; set_twist_topic[label="set_twist"];
     node [style=solid, shape=ellipse]; other_nodes;

     model -> model_plugin_link_node [style=dotted, dir=both];

     model_plugin_link_node -> pose_topic;
     model_plugin_link_node -> twist_topic;

     set_twist_topic -> model_plugin_link_node;
     set_pose_topic -> model_plugin_link_node;

     twist_topic -> other_nodes;
     pose_topic -> other_nodes;

     other_nodes -> set_pose_topic;
     other_nodes -> set_twist_topic;

     subgraph topics {
       rank="same"
       pose_topic
       set_pose_topic
       twist_topic
       set_twist_topic
     }
     subgraph gazebo {
       rank="same"
       label="Gazebo"

       model
     }
   }
|

.. _automatic_drive_node:

AutomaticDriveNode
------------------

The :py:class:`simulation.src.gazebo_simulation.src.automatic_drive.node.AutomaticDriveNode`
moves the car on the right side of the road.
It can be used instead of **KITcar_brain**.

.. admonition:: Launch

   .. prompt:: bash

      roslaunch gazebo_simulation automatic_drive.launch

The speed of the car can be modified by passing *speed:=...* as a launch parameter.
(Or by modifying the parameter with **rosparam** at runtime.)
