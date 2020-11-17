.. _drive_in_simulation:

Drive in Simulation
=====================

There are several components necessary to make the same code run in
simulation as on the real vehicle.
The following schema provides an oversimplified but instructive overview
of how the pieces are connected:

.. graphviz::
   :align: center
   :caption: Schema of the Simulation's Processes

   digraph Simulation {
     rankdir="LR";

     node [style=dotted, shape=box]; sensor_data [label="Sensor Data"]; car_model [label="Vehicle"];
     node [style=filled]; sensor_nodes [label="Sensor Nodes"]; vehicle_simulation_link [label="Vehicle Simulation Link"];
     node [style=dotted, shape=box]; brain [label="KITcar_brain"];vehicle_simulation[label="Vehicle Simulation"];

     subgraph cluster_gazebo {
       rank="same"
       label="Gazebo"

       sensor_data
       car_model
     }

     subgraph cluster_kitcar_gazebo_simulation {
        rank="same"
		    label = "kitcar-gazebo-simulation";

		    sensor_nodes;
        vehicle_simulation_link
	   }

     subgraph cluster_kitcar_ros {
       rank="same"
       label="kitcar-ros"

       brain
       vehicle_simulation
     }

     sensor_data -> sensor_nodes;
     vehicle_simulation_link -> car_model;

     sensor_nodes -> brain;
     vehicle_simulation -> vehicle_simulation_link;
   }
|

#. The simulation's pipeline is triggered when Gazebo generates new sensor data.
   E.g. Gazebo published a new simulated camera image.
#. This sensor data is then brought into the right format by the *Sensor Nodes*;
   e.g. the camera image is cropped and published on the topics,
   where **KITcar_brain** expects the data.
#. The **Vehicle Simulation** calculates the state of the car depending on **KITcar_brain**'s
   output.
#. The **Vehicle Simulation Link** moves the car according to the **Vehicle Simulation**'s
   output.

In a nutshell, this is how the car can drive within the simulation!

Without kitcar-ros
------------------

The schematic above is great to get an overview, but it is not enough to fully understand how kitcar-ros is integrated.
Firstly, all processes that are required to connect the car's code with the simulation are defined in :ref:`simulation_brain_link`.
The nodes used there are specifically designed for KITcar's code and must be adjusted to work with other code.

The two important questions are:

#. How to prepare the sensor data for the code that usually runs on other hardware?
#. How to realistically move the simulated car?

The answers to these questions are heavily dependent on the physical car.

1. Preparing the Car and Sensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the code requires only a camera image (or other sensor data that Gazebo can simulate), that's great.
:ref:`models` describes how the car model is defined and generated.
Otherwise, it might be required to write utility nodes that subscribe to Gazebo's output and
modify it the data. The :ref:`sensors` are such nodes.

2. Move the Car
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This part is probably more complicated.
If there's already a dynamic simulation capable of calculating the car's position and
velocity available, that's great.
A simple utility node that publishes the car's position and velocity, as described in
:ref:`model_plugin_link`, is enough to let the car drive in Gazebo.

For kitcar-ros, the :ref:`vehicle_simulation_link_node` propagates KITcar's vehicle
simulation to Gazebo.

Otherwise, a dynamic simulation must be created.
A simple idea is to just give the car's desired speed to Gazebo.
It will then integrate the speed over time and calculate the position on it's own.
