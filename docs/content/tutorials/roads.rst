:orphan:

.. _roads:

Roads
=====

Whenever the car's behavior is simulated, a road is necessary.

.. important::

   Roads are python scripts within ``simulation/models/env_db``.
   The name of the file is the road's name.

The repositories already comes with a few predefined roads.

.. program-output:: cd $KITCAR_REPO_PATH/kitcar-gazebo-simulation/ &&  tree -L 3 -I __pycache__ --dirsfirst simulation/models/env_db
   :shell:

The renderer expects to find py:attr:`road` of type \
:py:class:`simulation.utils.road.road.Road` within the road script.

This implies that:

.. note::

   A new road can be any script which defines a global road variable of type \
   :py:class:`simulation.utils.road.road.Road` within ``simulation/models/env_db``.

E.g. the file ``simulation/models/env_db/custom_road.py`` with content:

.. code-block:: python

   from simulation.utils.road.road import Road
   from simulation.utils.road.sections import *

   road = Road()

   road.append(StraightRoad())  # Replace with other road sections

creates a very simple straight road, called **custom_road**.

There are a number of different types of road sections which are explained in
:ref:`road_sections`.

default_road
------------

When the simulation is launched without any additional *road* parameter:

.. prompt:: bash

   roslaunch gazebo_simulation master.launch

the *default_road* is used.
It consists of multiple :py:mod:`simulation.utils.road.generator.road_sections` \
concatenated using a :py:class:`simulation.utils.road.road.Road`:

.. literalinclude:: ../../..//simulation/models/env_db/default_road.py
