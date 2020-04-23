Road Sections
=============

The road you have created in the :ref:`onboarding_simple_road` section sure looks nice but still is rather basic.
In this section, you are going to create a more complex world by adding further road sections.

.. tip::

   You can create and render the following road sections yourself by creating a custom road \
   ``simulation/models/env_db/example_road.py``:

   .. code-block:: python

      from road.road import Road
      from road.sections import *

      road = Road()

      road.append(StraightRoad())  # Replace with other road sections

   You can then generate the road

   .. prompt:: bash

      python3 -m generate_road example_road

   and start the simulation:

   .. prompt:: bash

      roslaunch gazebo_simulation master.launch road:=example_road


ParkingArea
----------------

.. figure:: road_examples/example_parking_area.jpg

As explained earlier, the *ParkingArea* can contain multiple children (*ParkingLot, ParkingSpot with ParkingObstacle and StartLine*).
This is an example on how to create a *ParkingArea*:

.. literalinclude::
   road_examples/example.py
   :language: python
   :linenos:
   :start-after: # - Beginning sphinx parking_area -
   :end-before: # - Ending sphinx parking_area -

It is a bit more complicated than a *StraightRoad*.

The *ParkingArea* takes three optional arguments, which are :py:attr:`start_line`, :py:attr:`left_lots`, and :py:attr:`right_lots`.
In the example, :py:attr:`start_line` is set to :py:func:`StartLine`. This creates a *StartLine* at the beginning of the *ParkingArea*.
If you do not want a StartLine, remove :py:attr:`start_line=StartLine()`. By default, :py:attr:`start_line` is :py:attr:`None`.
You can also discover this if you take a closer look at :py:class:`simulation.utils.road.section.parking_area`.
If you want to take this one step further, it is also possible to create a ParkingArea without any children; practically a StraightRoad.

The arguments :py:attr:`left_lots` and :py:attr:`right_lots` expect a list of *ParkingLots*. In this example two lots are created on the left and one is on the right side.
In the first lot on the left side, two *ParkingSpots* are placed. As you already know, *ParkingSpots* can have three different types.
The first spot in this example is occupied by a *ParkingObstacle*, the second is blocked, i.e. it looks like an X.

The second lot on the left looks different. You can also specify a :py:attr:`length` and an :py:attr:`opening_angle` for a *ParkingLot*.
Here, the start is set to two meters from the beginning of the *ParkingArea*. If you do not specify the start argument (like in the first lot) it is set to zero.
The opening angle is set to 40 degrees; the default is 60 degrees.
For the first spot in this lot, no arguments are given and thus it's kind is ParkingSpot.FREE and there's no obstacle placed inside.
This is the default behavior for a *ParkingSpot*.

.. Caution::

    Be careful: it is possible to place an obstacle on a free spot.
    The rendered road will look perfectly fine but it can cause problems in automatic driving tests because on a free spot no obstacle is expected.

Moving to the single lot on the right side, you can see the third optional argument for a *ParkingLot*.
It is called :py:attr:`depth` and controls the depth (along the y-axis) of a lot.
There is no length parameter because the length (along the x-axis) is calculated as the sum of all spots in one lot.
To change the size of a spot along the x-axis, simply specify a :py:attr:`width` parameter.
You can not set the depth of a spot because it is derived from the parent lot.

Intersection
------------

.. figure:: road_examples/example_intersection.jpg

On the previously generated road, there already was an intersection present. The two crossing roads intersect at a right angle.

.. literalinclude::
   road_examples/example.py
   :language: python
   :linenos:
   :start-after: # - Beginning sphinx intersection -
   :end-before: # - Ending sphinx intersection -

In this example, the crossing roads intersect at a 110-degree angle. The :py:attr:`turn` parameter causes an arrow drawn on the road to indicate the right turn.
The possible turn values are :py:attr:`RIGHT`, :py:attr:`LEFT` and :py:attr:`STRAIGHT`, the latter is the default.
The default size is 1.8 m and represents the length of each of the crossing roads.

ZebraCrossing
-------------

.. figure:: road_examples/example_zebra_crossing.jpg

The zebra crossing spans the entire length of this section. If no length argument is given, it defaults to 0.45 m.

.. literalinclude::
   road_examples/example.py
   :language: python
   :linenos:
   :start-after: # - Beginning sphinx zebra_crossing -
   :end-before: # - Ending sphinx zebra_crossing -

CircularArc
-----------

.. figure:: road_examples/example_arc.jpg

This section creates a circular arc pointing to the left (*LeftCircularArc*) and right (*RightCircularArc*).
This means instead of creating an arc with a negative radius to make it turn right the radius is always positive.
The two **mandatory** parametes for an arc are :py:attr:`radius` and :py:attr:`angle`.

.. literalinclude::
   road_examples/example.py
   :language: python
   :linenos:
   :start-after: # - Beginning sphinx left_arc -
   :end-before: # - Ending sphinx left_arc -

This example creates a circular arc to the left resulting in a 90-degree turn.

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
   * You can look at the road you have defined by generating it and starting  the simulation!

Don’t forget to commit and push your changes after completing the task!

.. note::

   By default, new roads are not included into git because we don't want to have everybody's roads in our repository!
   However, if you want to commit a road (like you should do here!). you can go into the roads folder (``simulation/models/env_db``) and execute:

   .. prompt:: bash

      git add -f <FILE_NAME>
