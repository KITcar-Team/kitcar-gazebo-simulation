############
ROS Packages
############

We already covered the basics of ROS in our `wiki`_.
In this part, we will explain the basic structure of our ROS packages.
They all share a basic structure that will be explained through the
:mod:`simulation.src.simulation_onboarding` package. This package is located at
``simulation/src/simulation_onboarding`` and will be used throughout the onboarding.

.. _wiki: https://wiki.kitcar-team.de/doku.php?id=onboarding:allgemein:ros_einfuehrung

.. Create the onboarding tree and hide ending and beginning of ouput with head and tail
   command
.. program-output::
   cd $KITCAR_REPO_PATH/kitcar-gazebo-simulation/ && tree -a -I __pycache__ --dirsfirst simulation/src/simulation_onboarding | head -n -2
   :shell:

****************
Launch directory
****************

Inside the ``launch`` directory are files which define what should be done when you execute
for example this command:

.. prompt:: bash

   roslaunch simulation_onboarding master.launch road:=<ROAD_NAME>

Let's break this command down:

- *simulation_onboarding* is the name of the ROS Package.
- *master.launch* is the name of the launch file that is launched by **roslaunch**.
- *road:=<ROAD_NAME>* is used to pass the launch file the argument **road** \
  with a value of *<ROAD_NAME*.

Let's take a look at ``master.launch``:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/launch/master.launch
   :language: xml
   :linenos:

As you can see, two arguments are defined and used within the launch file
``master.launch``. The ``onboarding_node.launch``
file is included and also launched. Let's take a look at it:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/launch/onboarding_node.launch
   :language: xml
   :linenos:

The *onboarding_node* gets executed with additional parameters loaded from ``topics.yaml``.

*****************
Message directory
*****************

ROS messages are defined inside the directory ``msg``. Each *\*.msg* file defines a new
message type. Learn more about ROS messages at http://wiki.ros.org/Messages. You can see an
example here:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/msg/Speed.msg
   :linenos:

*******************
Parameter directory
*******************

Each node has its own subdirectory inside the ``param`` directory.
In there is always a file called ``topics.yaml``.
It defines the topics which are published from the node.
Here is the ``topics.yaml``:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/param/onboarding/topics.yaml
   :language: yaml
   :linenos:

Additionally, the directory usually contains the ``default.yaml`` file; it defines parameters
used within the node:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/param/onboarding/default.yaml
   :language: yaml
   :linenos:

.. note::

   Using parameters in nodes creates programs with flexible behavior and
   code that does not depend on *magic-numbers*,
   i.e. numbers that are defined within the code and are very hard to read!

*****************
Scripts directory
*****************

The ``scripts`` directory contains python scripts that can start ROS nodes.

The ``scripts/onboarding_node`` is called when launching *onboarding_node.launch* which in
turn initializes an instance of *OnboardingNode*:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/scripts/onboarding_node
   :language: python
   :linenos:

Now you know how **roslaunch** uses python scripts to start nodes.

****************
Source directory
****************

The ``src`` directory is the heart of every ROS package. It contains the actual Python code
defining ROS nodes. Each node is defined within a subdirectory *(== python package)* within
``src``. Inside this subdirectory, you can find the node and sometimes additional Python
modules that add functionality to it. In this example, the onboarding node is defined in the
module ``node.py`` within the ``src/onboarding`` directory.

We will explain this code in the :ref:`next chapter<Node>`, but for integrity, here is the
file ``node.py``:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/src/onboarding/node.py
   :language: python
   :linenos:

***************
CMakeLists file
***************

The file ``CMakeLists.txt`` contains information for the compiler so it knows what has to be
done when it gets executed.

It should be executed in ``$KITCAR_REPO_PATH/kitcar-gazebo-simulation/simulation/`` with
the command:

.. prompt:: bash

    catkin_make

The file looks like this:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/CMakeLists.txt
   :language: bash
   :linenos:
   :lines: 4-6, 10-30, 51-59, 94-110, 169-

************
Package file
************

The file ``package.xml`` holds metadata about this ROS Package:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/package.xml
   :language: xml
   :linenos:

**********
Setup file
**********

The ``setup.py`` file tells *cmake* where the Python packages are located:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/setup.py
   :language: python
   :linenos:
   :start-after: # - Beginning sphinx -
