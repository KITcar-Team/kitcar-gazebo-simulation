.. _rosbags:

Simulated Rosbags
=================

Sometimes it might be useful to record rosbags of the simulation.
For example, to generate sets of simulated images, as are necessary to train the
:py:mod:`simulation.utils.machine_learning.cycle_gan`, rosbags are recorded and the
images extracted afterward.

record_random_drive.launch
--------------------------

Within the ROS package :ref:`simulation_groundtruth` a launch file to record the car automatically
driving along the road is defined:

.. note:: The simulation can be recorded by running:

   .. code-block:: shell

      roslaunch gazebo_simulation record_random_drive.launch rosbag_name:={NAME OF ROSBAG}

Just as the name suggests, the car will drive randomly on both sides of the road.
However, the optional parameter *randomize_path:={true/false}* can be used to fixate the car
to the middle of the right lane.

There is also a convenience script
``simulation/utils/machine_learning/data/record_simulated_rosbag.py`` that can be used to
easily record rosbags of the car driving on a number of roads at once.


Extract Information
-------------------

There are also multiple scripts defined in ``simulation/utils/machine_learning/data/`` to
extract the information from rosbags without replaying manually:

#. ``rosbag_to_images.py``: Extract images from a rosbag (or multiple rosbags).
#. ``rosbag_to_labels.py``: Extract labels published by the `label_camera_node` from a rosbag.

.. note:: The scripts can be executed with

   .. prompt:: bash

      python3 -m simulation.utils.machine_learning.data.{NAME OF THE SCRIPT}

   The flag *--help* reveals information about individual parameters and their usage.
