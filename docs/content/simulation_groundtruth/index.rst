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

The :mod:`simulation.src.simulation_groundtruth.src.groundtruth.node` provides \
a number of ROS services through which the groundtruth can be accessed by other ROS nodes.

.. note:: The node can be launched with:

   .. code-block:: shell

      roslaunch simulation_groundtruth groundtruth_node.launch road:={NAME_OF_ROAD}

   Parameters to modify the road and other properties can be found in \
   *param/groundtruth/default.yaml*

When started with the name of a road as a parameter the node uses the \
:mod:`simulation.utils.road_generation` to load a list of all road sections.
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
