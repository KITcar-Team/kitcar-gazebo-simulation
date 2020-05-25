Road Sections
=============

The road you have created in the :ref:`onboarding_simple_road` section sure looks nice but still is rather basic.
In this section, you are going to create a more complex world by adding further road sections.

.. tip::

   You can create and render the following road sections yourself by creating a custom road \
   ``simulation/models/env_db/example_road.py``:

   .. code-block:: python

      from simulation.utils.road.road import Road
      from simulation.utils.road.sections import *

      road = Road()

      road.append(StraightRoad())  # Replace with other road sections

   Then start the simulation:

   .. prompt:: bash

      roslaunch gazebo_simulation master.launch road:=example_road


.. include:: ../tutorials/road_sections.rst
   :start-after: .. onboarding_start
   :end-before: .. onboarding_end

Create Road
-----------

While the above introduction to the different road section is not complete and does neither explain all possible parameters which can be set
nor introduces all road sections available, it is sufficient for completing the following task.
If you want more information on the road sections just have all look at :ref:`simulation.utils.road.sections` the source code.

.. admonition:: Your Task

    Create a new road with the following sections:

    #. straight road: length 1 m
    #. parking area: length 3.5 m

        #. left lot: start after 0.5 m, depth 0.5 m, three parking spots
        #. right lot: start after 1.5 m, two spots with a width of 70 cm

    #. circular arc to the left: radius 2 m, angle 90 degrees
    #. zebra crossing
    #. intersection: turn to the left, size 3 m
    #. circular arc to the right: radius 1.5 m, angle 90 degrees
    #. circular arc to the left: radius 2.5 m, angle 60 degrees
    #. straight road: length 80 cm
    #. circular arc to the left: radius 2.5 m, angle 120 degrees
    #. intersection: size 3.2 m, angle 110 degrees, go straight
    #. straight road: length 1.66 m
    #. circular arc to the left: radius 1 m, angle 90 degrees
    #. straight road: length 1.25 m

    The road should form a closed loop.

    .. figure:: road_examples/onboarding_complex_road.jpg

.. hint::

   * Do not forget to import necessary sections.
   * You can look at the road you have defined by starting the simulation!
   * All angles must be specified in radians!

Don’t forget to commit and push your changes after completing the task!

.. note::

   By default, new roads are not included into git because we don't want to have everybody's roads in our repository!
   However, if you want to commit a road (like you should do here!). you can go into the roads folder (``simulation/models/env_db``) and execute:

   .. prompt:: bash

      git add -f <FILE_NAME>
