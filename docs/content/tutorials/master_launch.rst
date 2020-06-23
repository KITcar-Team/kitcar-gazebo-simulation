.. _master_launch:

master.launch
=============

The most important launch within **kitcar-gazebo-simulation** is the **master.launch**
file located at ``simulation/src/gazebo_simulation/launch/master.launch``.
It includes all parts necessary to start the simulation,
but also allows to individually modify which parts of the simulation are started.

Example: Start RVIZ with Road's Groundtruth
-------------------------------------------

One of the simulation's great strengths is that detailed information of the simulated world is available.
The following example shows how to visualize the groundtruth information to get a better
understanding of where the car perceives lanes and other information of the road.

.. admonition:: Launch

   Passing *rviz:=true* and *groundtruth:=true* along to the **master.launch** starts
   the simulation with an instance of the :ref:`groundtruth_node` and RVIZ.

   .. prompt:: bash

      roslaunch gazebo_simulation master.launch groundtruth:=true rviz:=true

This will open up RVIZ and after starting **kitcar-ros** detected lines and other objects seen by the car are displayed:

.. raw:: html

   <video width="100%" class="video-background" autoplay loop muted playsinline>
     <source src="../rviz_master_launch.mp4" type="video/mp4">
   Your browser does not support the video tag.
   </video>

|

Configurations
--------------

There are several other parameters available when launching **master.launch**.
Take a look at the actual launch file for more details:

.. literalinclude:: ../../../simulation/src/gazebo_simulation/launch/master.launch
   :language: xml
