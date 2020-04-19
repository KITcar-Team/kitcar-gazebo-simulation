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

When the simulation is launched without any additional *road* parameter:

.. prompt:: bash

   roslaunch gazebo_simulation master.launch

the *default_road* is used.
It consists of multiple :py:mod:`simulation.utils.road.generator.road_sections` \
concatenated using a :py:class:`simulation.utils.road.road.Road`:

.. literalinclude:: ../../..//simulation/models/env_db/default_road.py

As described in :ref:`getting_started`, the *default_road* must be generated before launching:

.. prompt:: bash

   python3 -m generate_road default_road

The :py:mod:`simulation.utils.generate_road` script imports the *<ROAD_NAME>.py* within \
``simulation/models/env_db/``.
It then expects to find the python variable **road** of type \
:py:class:`simulation.utils.road.road.Road` within the road script.

This implies that:

.. note::

   A new road can be any script which defines a global road variable of type \
   :py:class:`simulation.utils.road.road.Road` within ``simulation/models/env_db``.

There are a number of different types of road sections which are explained in
:ref:`road_sections`.
