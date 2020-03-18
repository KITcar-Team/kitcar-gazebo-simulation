.. KITcar Simulation documentation master file, created by
   sphinx-quickstart on Sun Feb 16 00:00:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to KITcar's Simulation
==============================

This simulation, developed and used by KITcar, a group of students participating in the annual CaroloCup_, can generate CaroloCup-Cup roads and uses the Gazebo_ simulator to enable vehicles with visual sensors to drive on the generated roads. 

Most components of the simulation have been written in Python3.6_ and use ROS_. ROS-Nodes are used to achieve high code modularity and ROS-Topics to achieve human readable interactions between code components. 

The most basic, but nevertheless most powerful feature of this simulation is to provide a world which allows to generate sensor data and provides interfaces to move your vehicle around freely.

What's needed for that?

* a road and
* a vehicle model and
* a simulation for the movement of the vehicle.

This package allows to generate a wide range of roads and to easily integrate a Gazebo_ model with camera sensors.
The actual movement (pose and twist) of the vehicle is not simulated, but must be done externally and can be integrated by writing a Gazebo model plugin. 

See :ref:`getting_started` for a tutorial on how to set up the Gazebo_ simulation.

Components
==================
A number of components are needed to run the simulation. The following list are the ROS-packages used:

.. toctree::
   :maxdepth: 2
 
   src/gazebo_simulation/index.rst
   src/simulation_brain_link/index.rst


.. _CaroloCup: https://wiki.ifr.ing.tu-bs.de/carolocup/news
.. _Gazebo: http://gazebosim.org
.. _ROS: https://www.ros.org/
.. _Python3.6: https://www.python.org/downloads/

The utils package contains useful tools to make life easier:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   utils/geometry/index.rst
   utils/ros_base/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
