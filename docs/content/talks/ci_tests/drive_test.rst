Drive Test
==========

A ROS-test that:

#. Launches the simulation with the complete evaluation pipeline,
#. then sets the new mission mode to start the car and
#. listens to the referee for updates, whether the car

   * finishes: Test succeeds
   * or makes a mistake: Test fails


See :ref:`drive_test` for more details.

.. admonition:: Run

   The test can be run with:

   .. prompt:: bash

      rostest simulation_evaluation drive.test road:=${ROAD_NAME} mission_mode:={ 1 or 2 }
