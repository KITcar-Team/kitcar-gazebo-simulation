.. KITcar Simulation documentation master file, created by
   sphinx-quickstart on Sun Feb 16 00:00:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to KITcar's Simulation
==============================

This simulation, developed and used by KITcar, a group of students participating in the anual CaroloCup_, can generate CaroloCup-Cup roads and uses the Gazebo_ simulator to enable vehicles with visual sensors to drive on the generated roads. 

Most components of the simulation have been written in Python3.6_ and use ROS_. ROS-Nodes are used to achieve high code modularity and ROS-Topics to achieve human readable interactions between code components. 

.. _CaroloCup: https://wiki.ifr.ing.tu-bs.de/carolocup/news
.. _Gazebo: http://gazebosim.org
.. _ROS: https://www.ros.org/
.. _Python3.6: https://www.python.org/downloads/

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
