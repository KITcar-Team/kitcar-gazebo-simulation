.. _onboarding_simple_road:

Road
============
This section is an introduction to the construction of routes for our simulation environment.
The routes for our simulation, which we call **roads**, consist of different sections.
A single road is made up of different road sections which are placed consecutively.
All length specifications are in meters.

Road sections
-------------
Available road sections are:

- :py:class:`simulation.utils.road.sections.road_section.RoadSection`: This is the abstract parent to all other road sections.
- :py:class:`simulation.utils.road.sections.straight_road.StraightRoad`: Simple straight road with a default length of 1 m.
- :py:class:`simulation.utils.road.sections.circular_arc.LeftCircularArc` and :py:class:`simulation.utils.road.sections.circular_arc.RightCircularArc`: Circular arcs take an angle and radius
- :py:class:`simulation.utils.road.sections.bezier_curve.QuadBezier` and :py:class:`simulation.utils.road.sections.bezier_curve.CubicBezier`: Bézier curves are parametric curve which is defined by a number of control points.
- :py:class:`simulation.utils.road.sections.intersection.Intersection`: Four-way intersection; by default the crossing roads are perpendicular to each other, but this angle can be freely modified. It is also possible to close up endings to get a *T*-junction for example.
- :py:class:`simulation.utils.road.sections.zebra_crossing.ZebraCrossing`: A cross walk; the default length is 0.45 m.
- :py:class:`simulation.utils.road.sections.parking_area.ParkingArea`: The *ParkingArea* can have  multiple *ParkingLots* on the left and right side of the road. According to the rules of the Carolo-Cup *ParkingLots* on the right side are meant for parallel parking, those on the left require perpendicular parking. *ParkingLots* consist of *ParkingSpots* which can have three different states:

    - *FREE:* please park here
    - *BLOCKED:* spots which are blocked are marked with a cross
    - *OCCUPIED:* spots occupied by *ParkingObstacles* which are simple cuboids

     Another important part is the *StartLine* which is also available inside the *ParkingArea*. The *StartLine* defines both the beginning of a road and the start of a *ParkingArea*.

*StaticObstacles* can be placed on every *RoadSection*. Every road section has a middle, right, and left line.
Lines can be dashed, solid or missing. The width of a single lane is 0.4 m thus making the road 0.8 m in width.
The origin of every road is at the beginning of the first middle line.
The x-Axis points in the driving direction (along the middle line), the y-Axis is defined pointing towards the left line.
These axes are also shown in *gazebo* (x-axis: red, y-axis: green).

Road directory
--------------
Roads are defined in a single python file. All of these road files are located in ``simulation/models/env_db/``.
After rendering a road file called ``road_name.py`` a ``.road_name`` folder is created in the same directory.
Amongst other things, this hidden folder contains pictures of the rendered road which are later used in the simulation.

.. tip::
    Hidden files or directories can be recognized by the dot (.) before their file or directory name.
    By default, they are not shown to the user. In most file managers, you can use the ``Ctrl+H`` keyboard shortcut to
    toggle between displaying or hiding hidden files.
    In a terminal you can use the `ls` command to display all files, including the hidden ones:

    .. prompt:: bash

        ls -a


Simple road
-----------
Inside the road directory, there is a road file called ``onboarding_simple.py``.
It is a very simple road only containing *StraightRoads* and a basic right-angle intersection.
Let’s take a look at ``onboarding_simple.py``:

.. literalinclude::
   ../../../simulation/models/env_db/onboarding_simple.py
   :language: python
   :linenos:

.. testsetup::

   from simulation.utils.road.road import Road  # Definition of the road class
   from simulation.utils.road.sections import StraightRoad
   from simulation.utils.road.sections import Intersection
   road = Road()
   road_section = StraightRoad()  # Just a dummy

Lines 3 to 5 are imports for the abstract road class and the used road sections.
In line 8 a *Road* called road is constructed and in the following lines different road sections are added to this road:

>>> road.append(road_section)

A *StraightRoad* with the default length of 1 m can be added by calling:

>>> StraightRoad()
StraightRoad(id=0, transform=Transform(translation=(0.0, 0.0, 0.0),rotation=0.0 degrees), is_start=False, left_line_marking='solid', middle_line_marking='dashed', right_line_marking='solid', obstacles=[], traffic_signs=[], surface_markings=[], length=1)

If you want to pass a different length add the argument :py:attr:`length` in the constructor. For example a 2 m long *StraightRoad*:

>>> StraightRoad(length=2)
StraightRoad(id=0, transform=Transform(translation=(0.0, 0.0, 0.0),rotation=0.0 degrees), is_start=False, left_line_marking='solid', middle_line_marking='dashed', right_line_marking='solid', obstacles=[], traffic_signs=[], surface_markings=[], length=2)


In the next section you are going to learn how to start the simulation with a custom road.
