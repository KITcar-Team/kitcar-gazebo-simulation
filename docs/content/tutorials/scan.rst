:orphan:

.. _scan::

Scan Real Roads
===============

The :py:class:`simulation.src.simulation_groundtruth.src.scan.node.ScanNode` can read the
car's position from rosbags to create a simulated road.

The node listens to the ROS topic */tf* and assumes that the car is always in the middle of
the right lane. *This requires a well suited rosbag.*

Scanning new Roads
------------------

Adding new roads requires multiple (easy) steps:

#. Start ROS:

   .. prompt:: bash

     roscore

#. Ensure that ROS uses time from the rosbag:

   .. prompt:: bash

     rosparam set /use_sim_time true

#. Start playing the rosbag with the rosbag's clock:

   .. prompt:: bash

     rosrun rosbag play ROSBAG_NAME --clock

   (You can always pause and play using the SPACE key.)
#. Then start the scan node to generate a *.yaml* file with information about the part
   of the road played in the rosbag:

   .. prompt:: bash

     roslaunch simulation_groundtruth scan_node.launch file_path:=PATH_TO_RESULTING_FILE

   When the node is stopped, a file containing the segment of the road the car drove while
   the node was running, is created.
#. Add the scanned road section to an existing road:

   .. code-block:: python

      from simulation.utils.road.road import Road
      from simulation.utils.road.sections import CustomSection

      road = Road()
      section = CustomSection.from_yaml(FILEPATH)
      road.append(section)

   *Currently, intersections can not be recorded like this. The best practice is to simply
   record the road before and after the intersection separately.*

Example: Carolo-Cup 2020 Training Loop
--------------------------------------

A road that has been recreated from a rosbag is our training road at the Carolo-Cup in 2020.
*Some additional road sections and minor adjustments were required to fit all parts
together.*

.. literalinclude::
   ../../../simulation/models/env_db/real_roads/cc20_train.py
   :language: python
   :linenos:
