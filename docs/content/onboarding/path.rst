Path
====

Now that you have made your road more complex, \
you also have to adjust where the car drives as well.
For that, you will need to calculate the position and \
orientation of the car along the right side of the road.

Geometry Package
----------------

The purpose of this Onboarding page is for you to get to know our \
:py:mod:`simulation.utils.geometry`-package.
The package contains implementations of geometrical objects that provide powerful, \
yet easy to use functionality.
When reading through our source code you will encounter these classes everywhere.

To complete the following task you will benefit from reading into
:py:mod:`simulation.utils.geometry`'s documentation \
to get a better understanding of these classes.

For now, we will just take a look at three of these classes.

.. tip::

  You can follow along with these tutorials by opening a Python interpreter shell in a
  terminal:

  .. prompt:: bash

     python3

Point
^^^^^

The :py:class:`simulation.utils.geometry.point.Point` can be initialized \
from multiple coordinates:

>>> from simulation.utils.geometry import Point
>>> p = Point(2,3,0)
>>> p
Point(2.0, 3.0, 0.0)

Because all geometry classes inherit from \
`Shapely <https://pypi.org/project/Shapely/>`_ classes, Shapely's functionality can be \
used as well.

Line
^^^^

We will now use the https://shapely.readthedocs.io/en/latest/manual.html#object.buffer
function in combination with the :py:class:`simulation.utils.geometry.line.Line` to create \
a circle:

>>> p = Point(0,0)
>>> shapely_poly = p.buffer(1, resolution=32)
>>> shapely_poly  # doctest: +SKIP
<shapely.geometry.polygon.Polygon object at 0x7f8d159ecf40>

The :py:attr:`shapely_poly` is a shapely polygon in the shape of a circle.
(Shapely's documentation about :py:attr:`resolution=32`: *The optional resolution argument determines the number of segments used to approximate a quarter circle around a point*.)

Let's create a line from the polygon's outside:

>>> from simulation.utils.geometry import Line
>>> circle = Line(shapely_poly.exterior.coords)
>>> circle.get_points()
[Point(1.0, 0.0, 0.0), Point(0.99879546, -0.04906767, 0.0), Point(0.99518473, -0.09801714, 0.0), Point(0.98917651, -0.14673047, 0.0), Point(0.98078528, -0.19509032, 0.0), Point(0.97003125, -0.24298018, 0.0), Point(0.95694034, -0.29028468, 0.0), Point(0.94154407, -0.33688985, 0.0), Point(0.92387953, -0.38268343, 0.0), Point(0.90398929, -0.42755509, 0.0), Point(0.88192126, -0.47139674, 0.0), Point(0.85772861, -0.51410274, 0.0), Point(0.83146961, -0.55557023, 0.0), Point(0.80320753, -0.5956993, 0.0), Point(0.77301045, -0.63439328, 0.0), Point(0.74095113, -0.67155895, 0.0), Point(0.70710678, -0.70710678, 0.0), Point(0.67155895, -0.74095113, 0.0), Point(0.63439328, -0.77301045, 0.0), Point(0.5956993, -0.80320753, 0.0), Point(0.55557023, -0.83146961, 0.0), Point(0.51410274, -0.85772861, 0.0), Point(0.47139674, -0.88192126, 0.0), Point(0.42755509, -0.90398929, 0.0), Point(0.38268343, -0.92387953, 0.0), Point(0.33688985, -0.94154407, 0.0), Point(0.29028468, -0.95694034, 0.0), Point(0.24298018, -0.97003125, 0.0), Point(0.19509032, -0.98078528, 0.0), Point(0.14673047, -0.98917651, 0.0), Point(0.09801714, -0.99518473, 0.0), Point(0.04906767, -0.99879546, 0.0), Point(0.0, -1.0, 0.0), Point(-0.04906767, -0.99879546, 0.0), Point(-0.09801714, -0.99518473, 0.0), Point(-0.14673047, -0.98917651, 0.0), Point(-0.19509032, -0.98078528, 0.0), Point(-0.24298018, -0.97003125, 0.0), Point(-0.29028468, -0.95694034, 0.0), Point(-0.33688985, -0.94154407, 0.0), Point(-0.38268343, -0.92387953, 0.0), Point(-0.42755509, -0.90398929, 0.0), Point(-0.47139674, -0.88192126, 0.0), Point(-0.51410274, -0.85772861, 0.0), Point(-0.55557023, -0.83146961, 0.0), Point(-0.5956993, -0.80320753, 0.0), Point(-0.63439328, -0.77301045, 0.0), Point(-0.67155895, -0.74095113, 0.0), Point(-0.70710678, -0.70710678, 0.0), Point(-0.74095113, -0.67155895, 0.0), Point(-0.77301045, -0.63439328, 0.0), Point(-0.80320753, -0.5956993, 0.0), Point(-0.83146961, -0.55557023, 0.0), Point(-0.85772861, -0.51410274, 0.0), Point(-0.88192126, -0.47139674, 0.0), Point(-0.90398929, -0.42755509, 0.0), Point(-0.92387953, -0.38268343, 0.0), Point(-0.94154407, -0.33688985, 0.0), Point(-0.95694034, -0.29028468, 0.0), Point(-0.97003125, -0.24298018, 0.0), Point(-0.98078528, -0.19509032, 0.0), Point(-0.98917651, -0.14673047, 0.0), Point(-0.99518473, -0.09801714, 0.0), Point(-0.99879546, -0.04906767, 0.0), Point(-1.0, -0.0, 0.0), Point(-0.99879546, 0.04906767, 0.0), Point(-0.99518473, 0.09801714, 0.0), Point(-0.98917651, 0.14673047, 0.0), Point(-0.98078528, 0.19509032, 0.0), Point(-0.97003125, 0.24298018, 0.0), Point(-0.95694034, 0.29028468, 0.0), Point(-0.94154407, 0.33688985, 0.0), Point(-0.92387953, 0.38268343, 0.0), Point(-0.90398929, 0.42755509, 0.0), Point(-0.88192126, 0.47139674, 0.0), Point(-0.85772861, 0.51410274, 0.0), Point(-0.83146961, 0.55557023, 0.0), Point(-0.80320753, 0.5956993, 0.0), Point(-0.77301045, 0.63439328, 0.0), Point(-0.74095113, 0.67155895, 0.0), Point(-0.70710678, 0.70710678, 0.0), Point(-0.67155895, 0.74095113, 0.0), Point(-0.63439328, 0.77301045, 0.0), Point(-0.5956993, 0.80320753, 0.0), Point(-0.55557023, 0.83146961, 0.0), Point(-0.51410274, 0.85772861, 0.0), Point(-0.47139674, 0.88192126, 0.0), Point(-0.42755509, 0.90398929, 0.0), Point(-0.38268343, 0.92387953, 0.0), Point(-0.33688985, 0.94154407, 0.0), Point(-0.29028468, 0.95694034, 0.0), Point(-0.24298018, 0.97003125, 0.0), Point(-0.19509032, 0.98078528, 0.0), Point(-0.14673047, 0.98917651, 0.0), Point(-0.09801714, 0.99518473, 0.0), Point(-0.04906767, 0.99879546, 0.0), Point(-0.0, 1.0, 0.0), Point(0.04906767, 0.99879546, 0.0), Point(0.09801714, 0.99518473, 0.0), Point(0.14673047, 0.98917651, 0.0), Point(0.19509032, 0.98078528, 0.0), Point(0.24298018, 0.97003125, 0.0), Point(0.29028468, 0.95694034, 0.0), Point(0.33688985, 0.94154407, 0.0), Point(0.38268343, 0.92387953, 0.0), Point(0.42755509, 0.90398929, 0.0), Point(0.47139674, 0.88192126, 0.0), Point(0.51410274, 0.85772861, 0.0), Point(0.55557023, 0.83146961, 0.0), Point(0.5956993, 0.80320753, 0.0), Point(0.63439328, 0.77301045, 0.0), Point(0.67155895, 0.74095113, 0.0), Point(0.70710678, 0.70710678, 0.0), Point(0.74095113, 0.67155895, 0.0), Point(0.77301045, 0.63439328, 0.0), Point(0.80320753, 0.5956993, 0.0), Point(0.83146961, 0.55557023, 0.0), Point(0.85772861, 0.51410274, 0.0), Point(0.88192126, 0.47139674, 0.0), Point(0.90398929, 0.42755509, 0.0), Point(0.92387953, 0.38268343, 0.0), Point(0.94154407, 0.33688985, 0.0), Point(0.95694034, 0.29028468, 0.0), Point(0.97003125, 0.24298018, 0.0), Point(0.98078528, 0.19509032, 0.0), Point(0.98917651, 0.14673047, 0.0), Point(0.99518473, 0.09801714, 0.0), Point(0.99879546, 0.04906767, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)]

We now have a complete circle as a line.
However, if we only want the first half of the circle, we can simply get the line's points as a list and create a new line from the first half of the points:

>>> points = circle.get_points()
>>> half_circle = Line(points[:len(points) // 2])
>>> half_circle.get_points()[-1]
Point(-1.0, -0.0, 0.0)

We can also create a new line by adding two lines:

>>> straight_line = Line([Point(-1,0),Point(-1,-1)])
>>> half_oval = half_circle + straight_line
>>> half_oval.get_points()[-5:]
[Point(-0.99518473, -0.09801714, 0.0), Point(-0.99879546, -0.04906767, 0.0), Point(-1.0, -0.0, 0.0), Point(-1.0, 0.0, 0.0), Point(-1.0, -1.0, 0.0)]

Another neat feature of lines is \
:py:func:`simulation.utils.geometry.line.Line.interpolate_pose`.
It allows you to approximate the pose at any point on the line.
E.g. if you know that the car travels on the :py:attr:`half_circle` \
and you know that it has travelled 1 meter so far, \
you can get the approximate position and orientation as a pose:

>>> half_circle.interpolate_pose(arc_length = 1)
Pose(position=(0.540066444297505, -0.8412872782351057, 0.0),orientation= -147.6563 degrees)

Transform
^^^^^^^^^

Defining circles and lines is neat, but it becomes cumbersome, \
if you have no way of moving them around.
The :py:class:`simulation.utils.geometry.transform.Transform` does just that.
You can use it to translate and rotate all other geometry classes, through simple multiplication:

>>> import math
>>> from simulation.utils.geometry import Transform, Point, Line
>>> tf = Transform(Point(1, 1, 0), math.pi / 2)  # Rotate around (0,0) by 90 degrees and shift by x=1, y=1.
>>> tf * Point(4, 2)
Point(-1.0, 5.0, 0.0)
>>> long_line = Line([Point(0, 0), Point(10, 0)])
>>> tf * long_line
Line([Point(1.0, 1.0, 0.0), Point(1.0, 11.0, 0.0)])

As you can see, :py:class:`simulation.utils.geometry.transform.Transform` rotates and then translates other geometry objects.
What if you want to translate before rotating?
Another great strength of transforms is, that they can be multiplied as well:

>>> rotate = Transform([0, 0], math.pi / 2)
>>> translate = Transform([1, 1], 0)
>>> translate * rotate
Transform(translation=(1.0, 1.0, 0.0),rotation=90.0 degrees)
>>> rotate * translate
Transform(translation=(-0.9999999999999999, 1.0, 0.0),rotation=90.0 degrees)

When multiplying two transforms, the product is another transform, \
that is equivalent to the right transform first and then the left one.

Drive on the Road
-----------------

With the explanations above and possibly reading a bit through our documentation,
you are prepared to tackle the last, but also the hardest task of the Onboarding:

.. admonition:: Your Task

   Modify the onboarding node to make the car drive on the new road you've created in the last part.

    .. figure:: road_examples/onboarding_result.gif

.. hint::

   We are aware, that this last task is not easy.
   Here are a few hints that you can, but not don't have to use:

   * Take a look at the individual sections that you've used to create the road.
     Try to figure out, what the middle line of the individual road section would be
     and then just add the middle lines together:

     >>> complete_middle_line = middle_line_1 + middle_line_2 + ...  # doctest: +SKIP

   * Once you know the middle line of the road you can use :py:func:`simulation.utils.geometry.line.Line.parallel_offset` to get the middle of the right lane (*where the car should drive*).

   * Think about how you could use the transform to connect multiple road sections together.

   * It might help (but is not necessary) to take a closer look at the source code of the road sections,
     maybe they have some useful properties ;)

   * Intersections have a :py:attr:`size` attribute that specifies their complete size.
     By default, it is 1.8 meters.

   * If you are not sure how to change the orientation of the car, \
     you should take another look at ``simulation/src/gazebo_simulation/msg/SetModelPose.msg``.
     There's a detailed description on how to use the message.

   * Once you have the pose, it's easy to get the orientation as a quaternion:

     >>> pose = half_circle.interpolate_pose(arc_length = 0)
     >>> pose.to_geometry_msg()  # doctest: +NORMALIZE_WHITESPACE
     position:
       x: 1.0
       y: 0.0
       z: 0.0
     orientation:
       x: -0.0
       y: -0.0
       z: -0.715730825283741
       w: 0.6983762494090524


   * Last but not least: Ask other team members for help!




Don't forget to commit and push your changes after completing the task!
