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


