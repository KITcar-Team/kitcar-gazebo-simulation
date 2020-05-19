.. KITcar Simulation documentation master file, created by
   sphinx-quickstart on Sun Feb 16 00:00:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _CaroloCup: https://wiki.ifr.ing.tu-bs.de/carolocup/news
.. _Gazebo: http://gazebosim.org
.. _ROS: https://www.ros.org/
.. _Python3.8: https://www.python.org/downloads/
.. _Github: https://www.github.com/...
.. _drive_sim_road_generation: https://github.com/tum-phoenix/drive_sim_road_generation
.. _TUM-Phoenix: https://www.mw.tum.de/phoenix/tum-phoenix-robotics/
.. _KITcar: https://kitcar-team.de

Welcome to KITcar's Simulation
==============================

This is the documentation of **kitcar-gazebo-simulation**; a ROS_ and Gazebo_ based simulation \
tool that can generate CaroloCup_ roads and simulate a car driving on the generated roads.

.. figure:: tutorials/resources/simulation_video.gif

   View of the Gazebo user interface displaying the default_road.

ROS_ topics allow to easily access sensor data and modify the car's position or speed. Thus, allowing to completely simulate the bevahior of the car.

.. figure:: tutorials/resources/simulation_camera.gif

   Front Camera Output of the Car.

.. note::

   There's currently no dynamics simulation. The position or speed of the car must be provided externally.

Public parts of the source code are available on Github.
The project was originally inspired by the drive_sim_road_generation_ \
created by TUM-Phoenix_ but significantly diverged since and \
grew into a completely independent project developed by KITcar_.

This documentation aims to enable an easy entry to using the simulation \
while still providing a "behind-the-scenes" explanation of the implementation.
First, there are several introductory pages, before diving into the different \
ROS packages and python packages used in the repository afterward.
The documentation concludes with a list of "mini"-talks that were held internally at KITcar \
to share knowledge and ensure common standards within the code.

.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   tutorials/installation
   tutorials/getting_started
   tutorials/master_launch
   tutorials/drive_in_simulation
   tutorials/roads.rst
   tutorials/road_sections.rst
   tutorials/models


.. toctree::
   :maxdepth: 2
   :caption: ROS Packages

   gazebo_simulation/index
   simulation_brain_link/index
   simulation_evaluation/index
   simulation_groundtruth/index


.. toctree::
   :maxdepth: 2
   :caption: Python Packages

   _source_files/simulation.utils.car_model
   _source_files/simulation.utils.geometry
   _source_files/simulation.utils.ros_base
   _source_files/simulation.utils.road
   _source_files/simulation.utils.urdf


.. toctree::
   :maxdepth: 1
   :caption: Mini - Talks

   talks/names_and_values/index.rst
   talks/inheritance/index.rst


Onboarding
----------

At KITcar_, we integrate new members into the team through the *Onboarding*-Challenge.
This is the **KITcar Simulation**'s Onboarding:

.. toctree::

   onboarding/index


