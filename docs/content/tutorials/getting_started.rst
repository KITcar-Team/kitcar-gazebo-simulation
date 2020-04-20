:orphan:

.. _getting_started:

Getting Started
=================

This page explains the first steps of using the simulation.
It only covers the very basics of what's happening behind the scenes.

Generate Roads
----------------

Roads are defined as python scripts within *simulation/models/env_db*.
Before starting the simulation with a road, a Gazebo model must be generated from the \
python script:

.. admonition:: Generate a Road

   .. prompt:: bash

      python3 -m generate_road <NAME_OF_ROAD>

The road name is equivalent to the name of the python script defining the road.
See :ref:`roads` for more details.

By default there are a number of predefined roads.
The **default_road** should be generated before continuing

.. prompt:: bash

   python3 -m generate_road default_road


Launch the Simulation
------------------------

The next step is to start the actual simulation:

.. admonition:: Launch the Simulation

   .. prompt:: bash

      roslaunch gazebo_simulation master.launch road:=<NAME_OF_ROAD>

The **default_road** can be started without the *road* argument:

.. prompt:: bash

   roslaunch gazebo_simulation master.launch

You should then see a road looking similar to the following:

.. figure:: resources/gazebo_simulation.jpg
