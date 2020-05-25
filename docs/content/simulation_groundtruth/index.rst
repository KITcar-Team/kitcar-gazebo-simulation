simulation_groundtruth
======================

The **groundtruth** ROS package enables other ROS nodes to access information \
about the current road.

.. toctree::
   :maxdepth: 1
   :caption: Packages and Modules

   ../_source_files/simulation.src.simulation_groundtruth.src.groundtruth

GroundtruthNode
---------------

The :mod:`simulation.src.simulation_groundtruth.src.groundtruth.node` has access to the
current road.
It uses this information not only to provide access to the roads groundtruth but also
render the road in Gazebo.

Groundtruth
^^^^^^^^^^^
The :mod:`simulation.src.simulation_groundtruth.src.groundtruth.node` provides \
a number of ROS services through which the groundtruth can be accessed by other ROS nodes.

.. note:: The node can be launched with:

   .. code-block:: shell

      roslaunch simulation_groundtruth groundtruth_node.launch road:={NAME_OF_ROAD}

   Parameters to modify the road and other properties can be found in \
   *param/groundtruth/default.yaml*

When started with the name of a road as a parameter the node uses the \
:mod:`simulation.utils.road` to load a list of all road sections.
The list of road sections is used to initialize a \
:mod:`simulation.src.simulation_groundtruth.src.groundtruth.groundtruth` object.
The groundtruth object can generate ROS msgs for specific requests \
(e.g. given a section id, return all obstacles).
When receiving a request on a ROS service, the groundtruth node passes the request \
to its groundtruth object and wraps the result in a service response object.

.. graphviz::

   digraph GroundtruthNode {
     rankdir="LR";

     node []; groundtruth [style=dotted]; groundtruth_node;
     node [shape=box]; section_service; lane_service; other_services [label="..."];
     node [shape=ellipse, style=dotted]; other_node; another_node;

     groundtruth -> groundtruth_node [style=dotted];
     groundtruth_node -> groundtruth [style=dotted];

     groundtruth_node -> section_service;
     groundtruth_node -> lane_service;
     groundtruth_node -> other_services;

     edge [style=dotted,dir=both]; section_service -> other_node;
     lane_service -> other_node;
     lane_service -> another_node;
     other_services -> another_node;

     subgraph services {
       rank="same"
       section_service
       lane_service
       other_services
     }
   }
|

Renderer
^^^^^^^^

Creating and displaying roads in Gazebo is an important part of the simulation.
The :py:class:`simulation.src.simulation_groundtruth.src.groundtruth.renderer.Renderer`
is started as a part of the
:py:class:`simulation.src.simulation_groundtruth.src.groundtruth.node.GroundtruthNode`.

The road is drawn onto the ground as an image.
I.e. all road lines are just an image that is displayed in Gazebo.
However, because roads can be very large, it is better to split up the road into equally
sized :py:class:`simulation.utils.road.renderer.tile.Tile`.

Additionally, there are some optimizations:

#. Only tiles with visible sections of the road are created
#. A road is rendered only once. If it is opened again, without modifying the road file,
   the previously rendered tiles are reused.

Obstacles and traffic signs must be created as well.
After the renderer has created it's groundplane,
:py:class:`simulation.src.simulation_groundtruth.src.groundtruth.object_controller.ObjectController`
spawns all obstacles and traffic signs.

Putting the pieces together; these are the steps taken to create and populate the Gazebo world:

#. Import the road from ``simulation/models/env_db/<ROAD_NAME>.py``,
#. check results from previous renderings are available, otherwise
#. distribute the road onto multiple equally sized tiles on the ground,
#. draw each tile and save the results to a file,
#. spawn the tiles in Gazebo, and
#. spawn obstacles and traffic signs with the ObjectController.

GroundtruthMockNode
-------------------

The :mod:`simulation.src.simulation_groundtruth.groundtruth.test.mock_node`  is a subclass \
of :mod:`simulation.src.simulation_groundtruth.src.groundtruth.node` providing the \
groundtruth of a number of simple roads.

.. note:: The node can be launched with:

   .. code-block:: shell

      roslaunch simulation_groundtruth groundtruth_mock_node.launch

   Parameters to modify the road and other properties can be found in \
   *param/groundtruth_mock/default.yaml*
