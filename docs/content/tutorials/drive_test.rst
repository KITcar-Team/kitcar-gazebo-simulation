.. _drive_test:

How to Write Tests Using the Simulation
=======================================

This drive test utility package located in :py:mod:`simulation.utils.drive_test` enables to
rapidly execute a lot of tests in an efficient way.


.. raw:: html

   <video width="100%" class="video-background" autoplay loop muted playsinline>
     <source src="../parking_drive_test.mp4" type="video/mp4">
   Your browser does not support the video tag.
   </video>

*Drive tests for a number of parking scenarios.*


Create your own Tests
---------------------

You can execute the test with this command:

.. code-block:: bash

    python -m simulation.utils.drive_test.run --config $KITCAR_REPO_PATH/kitcar-gazebo-simulation/docs/content/tutorials/resources/minimal_drive_test_config.yaml

This executes the Python package and gives it a path to a config that defines which tests should be executed.

After the test finishes, this table is displayed:

.. code-block::

    ╒════════════════╤════════════════════╤══════════════╤══════════════════════════╕
    │ desc           │ road               │   time_limit │ conclusion               │
    ╞════════════════╪════════════════════╪══════════════╪══════════════════════════╡
    │ Curvy road.    │ ci_roads/curves    │          300 │ Success (Expected: None) │
    ├────────────────┼────────────────────┼──────────────┼──────────────────────────┤
    │ Obstacle road. │ ci_roads/obstacles │          300 │ Failure (Expected: None) │
    ╘════════════════╧════════════════════╧══════════════╧══════════════════════════╛

The file ``docs/content/tutorials/resouces/minimal_drive_test_config.yaml`` which we pass to the drive_test looks like this:

.. literalinclude::
   ../../../docs/content/tutorials/resources/minimal_drive_test_config.yaml
   :language: yaml
   :linenos:

Take a closer look at this part:

.. literalinclude::
   ../../../docs/content/tutorials/resources/minimal_drive_test_config.yaml
   :language: yaml
   :linenos:
   :start-after: time_limit: 300

The list *tests* defines each test that should be executed during runtime.
The definiton starts with a hypen followed by a list of parameters.
Take a look at the table again. Every test has it's own row filled with these parameters.

As you may noticed, there are parameters inside the table that aren't inside the list *tests*.
There are default parameters which are applied to every test:

.. literalinclude::
   ../../../docs/content/tutorials/resources/minimal_drive_test_config.yaml
   :language: yaml
   :linenos:
   :start-after: table_column_max_width: 0
   :end-before: tests:

This improves readability inside the yaml file.
You can overwrite a default parameter by overwriting it inside the *tests* list!

Modifying the table
^^^^^^^^^^^^^^^^^^^

First, the header. The list *table_header* defines the header of the ouput table.

.. literalinclude::
   ../../../docs/content/tutorials/resources/minimal_drive_test_config.yaml
   :language: yaml
   :linenos:
   :end-before: table_column_max_width

And this line defines the width of the table. Setting this to 0 disables this feature.

.. literalinclude::
   ../../../docs/content/tutorials/resources/minimal_drive_test_config.yaml
   :language: yaml
   :linenos:
   :start-after: - conclusion
   :end-before: # Defaults get applied to every test

Valid parameters
^^^^^^^^^^^^^^^^

Until now we mentioned that you can pass some paramters. But which paramters can be passed?

There are some special paramters:

- *desc*: A description for the test. Just so you know what you are doing.
- *result*: If the test completed successfully.
- *must_succeed*: If you expected the test to complete successfully.
- *conclusion*: Shortcut for result and *must_succeed* in one box.

Every other valid parameter can be set inside the launch file
``simulation/src/simulation_evaluation/launch/test/drive.test``:

.. literalinclude::
   ../../../simulation/src/simulation_evaluation/launch/test/drive.test
   :language: xml
   :linenos:

Take a closer look at the beginning of this file. There you can see every other valid
parameter that can be passed to the script. You are also able to add your own of course!

.. literalinclude::
   ../../../simulation/src/simulation_evaluation/launch/test/drive.test
   :language: xml
   :linenos:
   :start-after: <launch>
   :end-before: <include file="$(find gazebo_simulation)/launch/master.launch">

Tests <=> Tickets
-----------------

Tests can also be allowed to fail. **What?? Why would one write a test and then allow it to fail?**

One can specify what the car **should be capable of** and thus define tests that
continuously check what the car is **currently capable of**.

The idea is simple. In traditional development *tickets* are used to plan and organize work.
At some point, it is decided what is currently good enough and what still has to be done.
In that sense, the tests are the same as writing *tickets*.

To specify a test as *allowed to fail*, just add the argument ``must_succeed: false``:

.. code-block:: yaml

  ...
  tests:
  - mission_mode: 1
    road: example_road
    must_succeed: false
  ...


.. admonition:: Example

  The following is a simple example to show how to test the car's behavior at multiple
  intersections closely after one another.
  The road is called ``ci_roads/double_intersection.py`` contains two intersections
  that can be modified using environment variables:

  .. literalinclude::
     ./road_examples/example_double_intersection_road.py

  A config that creates tests for all turn combinations looks like:

  .. literalinclude::
     resources/double_intersection_drive_test_config.yaml

  It can be executed with

  .. prompt:: bash

    python3 -m simulation.utils.drive_test.run --config $KITCAR_REPO_PATH/kitcar-gazebo-simulation/docs/content/tutorials/resources/double_intersection_drive_test_config.yaml

  The tests executed by the config **do not all succeed** and reveal that the car
  slightly overshoots when turning left:

  .. figure:: resources/double_intersection.jpg

  The result displayed when running the tests look are:

  .. code-block:: txt

    ╒════════════════════════════╤══════════════════════════════╤══════════════╤═════════════════════════════╕
    │ desc                       │ road                         │   time_limit │ conclusion                  │
    ╞════════════════════════════╪══════════════════════════════╪══════════════╪═════════════════════════════╡
    │ Right&right intersections. │ ci_roads/double_intersection │          300 │ Success (Expected: Success) │
    ├────────────────────────────┼──────────────────────────────┼──────────────┼─────────────────────────────┤
    │ Right&left intersections.  │ ci_roads/double_intersection │          300 │ Success (Expected: Success) │
    ├────────────────────────────┼──────────────────────────────┼──────────────┼─────────────────────────────┤
    │ Left&right intersections.  │ ci_roads/double_intersection │          300 │ Success (Expected: Success) │
    ├────────────────────────────┼──────────────────────────────┼──────────────┼─────────────────────────────┤
    │ Left&left intersections.   │ ci_roads/double_intersection │          300 │ Failure (Expected: Success) │
    ╘════════════════════════════╧══════════════════════════════╧══════════════╧═════════════════════════════╛

    Job has failed.

  The first left turns sometimes fail (as seen above) because the car slightly drives off the road after
  turning.
  One can now specify the failures by adding ``must_succeed: false`` to tests that fail.
  The result then looks like:

  .. code-block:: txt

    ╒════════════════════════════╤══════════════════════════════╤══════════════╤═════════════════════════════╕
    │ desc                       │ road                         │   time_limit │ conclusion                  │
    ╞════════════════════════════╪══════════════════════════════╪══════════════╪═════════════════════════════╡
    │ Right&right intersections. │ ci_roads/double_intersection │          300 │ Success (Expected: Success) │
    ├────────────────────────────┼──────────────────────────────┼──────────────┼─────────────────────────────┤
    │ Right&left intersections.  │ ci_roads/double_intersection │          300 │ Success (Expected: Success) │
    ├────────────────────────────┼──────────────────────────────┼──────────────┼─────────────────────────────┤
    │ Left&right intersections.  │ ci_roads/double_intersection │          300 │ Success (Expected: None)    │
    ├────────────────────────────┼──────────────────────────────┼──────────────┼─────────────────────────────┤
    │ Left&left intersections.   │ ci_roads/double_intersection │          300 │ Failure (Expected: None)    │
    ╘════════════════════════════╧══════════════════════════════╧══════════════╧═════════════════════════════╛

    All roads marked with must_succeed have been run successfully.


Why? Continuous Integration!
----------------------------

New code is hard to test. The more you rapidly you develop new features, the harder it is.
Sometimes certain changes in behavior are not predicted. Some bugs happen rarely.
ROS tests and Gtests have been used to test individual components for a long time.
These drive tests allow to **test situations** instead of components.
They are by no means a replacement but a great addition to existing component tests!


.. admonition:: Example

   Adding the example of above to an existing Gitlab-CI is very easy.
   The only requirement is that the right :ref:`docker` image is available.

   If this is the case, a job executing the above config is quite easy:

   .. code-block:: yaml

      # A simple CI job that runs the config above!
      test-double-intersection:
        image: git.kitcar-team.de:4567/kitcar/kitcar-gazebo-simulation/kitcar_ros_ci
        stage: test
        script:
        - python3 -m simulation.utils.drive_test.run --config $KITCAR_REPO_PATH/kitcar-gazebo-simulation/docs/content/tutorials/resources/double_intersection_drive_test_config.yaml

   And just like that, it is ensured that the car can handle double intersections
   for right turns and one can easily adapt the config above after fixing left
   turns!
