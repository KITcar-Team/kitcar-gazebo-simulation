.. _gazebo_simulation:

gazebo_simulation
=======================================================

ROS package to better communicate with Gazebo.

.. program-output:: cd $KITCAR_REPO_PATH/kitcar-gazebo-simulation && tree -a -I __pycache__ --dirsfirst simulation/src/gazebo_simulation
   :shell:


.. toctree::
   :maxdepth: 1
   :caption: Packages and Modules

   ../_source_files/simulation.src.gazebo_simulation.src.automatic_drive
   ../_source_files/simulation.src.gazebo_simulation.src.car_model
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


Model of the Car
----------------

An important part of the simulation is the vehicle it self.
It should be a realistic representation of the real car.
In our case we do not simulate the actual tire movement and basically
move a box around the simulated world.
This simplifies the model significantly!
Nevertheless, the sensors must be defined and should be as close
to their real specifications as possible.
Instead of calculating or estimating their position and calibrations,
the sensors specifications are extracted from the **car_specs**
ROS package within KITcar_brain.

Gazebo allows to define models using the `urdf <http://wiki.ros.org/urdf>`_ standard.
However, defining a model consisting of multiple parts and sensors is repetetive.
So instead of writing the **urdf** by hand, there's a Python script that generates it!

.. admonition:: Generate Dr. Drift

   Generate a new model definition of Dr. Drift and an updated calibration by running:

   .. prompt:: bash

      rosrun gazebo_simulation generate_dr_drift

Behind the scenes, there are multiple things going on:

1. The **car_specs** and **camera_specs** are loaded using
   :py:class:`simulation.src.gazebo_simulation.src.car_model.car_specs.CarSpecs`
   and :py:class:`simulation.src.gazebo_simulation.src.car_model.camera_specs.CameraSpecs`.
1. The model of Dr.Drift is defined as an :py:class:`simulation.utils.urdf.core.XmlObject`
   in :py:mod:`simulation.src.gazebo_simulation.src.car_model.dr_drift`.
   Using the **urdf** package allows to define classes for the individual parts of the vehicle.
1. The model, the car specs and the camera calibration is saved to
   ``simulation/src/gazebo_simulation/param/car_specs/dr_drift/``.


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

WorldPluginLinkNode
----------------------

The :ref:`world_plugin_link_node` is a \
`Gazebo world plugin <http://gazebosim.org/tutorials?tut=plugins_world&cat=write_plugin>`_
that allows to interact with worlds in Gazebo easily.
In particular, it allows to spawn and remove models from the world.

It can be attached to Gazebo worlds by adding

.. code-block:: xml

   <plugin filename="libworld_plugin_link.so" name="world_plugin_link" />

to the *model.sdf*.

When Gazebo loads a world with the *world_plugin_link*, a new instance of the world plugin \
link node is created.
The model plugin link node creates two subscribers:

- */simulation/gazebo/world/spawn_sdf_model* (string): \
  Receive sdf model definition.
- */simulation/gazebo/world/remove_model* (string): \
  Receive name of the model to be removed.

.. graphviz::
   :align: center
   :caption: Schema of the World Plugin Link

   digraph WorldPluginLink {

     node [style=dotted, shape=box]; world [label="Gazebo world"];
     node [style=solid, shape=ellipse]; world_plugin_link_node [label="world_plugin_link"];
     node [shape=box]; spawn_sdf [label="spawn_sdf_model"]; remove_model [label="remove_model"];
     node [style=solid, shape=ellipse]; other_nodes;

     world -> world_plugin_link_node [style=dotted, dir=both];

     world_plugin_link_node -> spawn_sdf[dir="back"];
     world_plugin_link_node -> remove_model[dir="back"];

     spawn_sdf -> other_nodes[dir="back"];
     remove_model -> other_nodes[dir="back"];

     subgraph gazebo {
       rank="same"
       label="Gazebo"

       world
     }
     subgraph topics {
       rank="same"
       spawn_sdf
       remove_model
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
