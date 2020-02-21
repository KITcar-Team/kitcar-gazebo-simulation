Geometry classes
=======================================================

This module contains python classes representing fundamental mathematical objects.
The classes were created to provide powerful mathematical operations while at the same time allow for an easy integration of ROS geometry_msgs_, numpy and schema.
The mathematical operations are inherited by subclassing the Shapely_ package and extending necessary functionality.

The following classes are available:


.. autoclass:: base.vector.Vector
   :members:
   :special-members:


.. autoclass:: base.point.Point
   :members:
   :special-members:


.. autoclass:: base.transform.Transform
   :members:
   :special-members:


.. autoclass:: base.pose.Pose
   :members:
   :special-members:


.. autoclass:: base.line.Line
   :members:
   :special-members:


.. autoclass:: base.polygon.Polygon
   :members:
   :special-members:

.. _Shapely: https://pypi.org/project/Shapely/
.. _geometry_msgs: http://wiki.ros.org/geometry_msgs
