.. _onboarding_subscriber:

ROS Subscriber
==============

You have already created a publisher that controls the car's position.
In this part, you will learn how to create a subscriber \
that can be used to interactively change the car's speed.


Speed Message
-------------

Because we want to set the speed of the car, we need a message \
that is suitable for transmitting the speed.
Messages can be defined in the ``msg/`` folder of a ROS package.
In this case, the message we need is already defined in \
``msg/Speed.msg``:

.. literalinclude:: ../../../simulation/src/simulation_onboarding/msg/Speed.msg

Additionally, a message needs to be added to the ``CMakeLists.txt``:

.. literalinclude:: ../../../simulation/src/simulation_onboarding/CMakeLists.txt
   :start-after: Add message definitions
   :end-before: Generate services in the

Now, we can import the message type

.. code-block:: python

   >>> from simulation_onboarding.msg import Speed

and initialize a subscriber within :py:func:`start` (and unregister in the :py:func:`stop`).

Create a Subscriber
-------------------

The publisher was initialized by adding

.. code-block:: python

   self.pose_publisher = rospy.Publisher(
       name="/simulation/gazebo/model/dr_drift/set_pose",
       data_class=SetModelPose,
       queue_size=1,
   )

to :py:func:`start`.

Similarly, you can add a subscriber with

.. code-block:: python

   self.speed _subscriber = rospy.Subscriber(
       name="/simulation/onboarding/speed",
       data_class=Speed,
       callback = self.speed_callback  # Function that gets called when a message is sent on the topic
   )

The subscriber requires a *callback*-parameter,
that must be a function which takes a message as a parameter;
i.e. with the above subscriber definition, \
you need to define the :py:func:`speed_callback` function:

.. code-block:: python

   def speed_callback(self, msg):
       """Receive new speed in message."""
       rospy.loginfo(f"Received speed message: {msg}")

Let's test if the subscriber receives messages.
You can launch your code with:

.. prompt:: bash

   roslaunch simulation_onboarding onboarding_node.launch

While your node is running, \
you can send messages to the speed subscriber from a second terminal with

.. prompt:: bash

   rostopic pub /simulation/onboarding/speed simulation_onboarding/Speed "speed: 5.0"

If everything works as expected, \
you will see a new line in the terminal the node is running::

   [INFO] [1586247506.869336] [/simulation_onboarding/onboarding_node]: Received speed message: speed: 5.0

.. tip::

   Many ROS commands are easy to use with autocompletion.
   E.g. instead of typing the complete command,
   you can just type **rostopic pub /** and hit **<TAB>** three times.


Change the Car's Speed
----------------------

With the subscriber receiving messages you can now go ahead and start your next task:


.. admonition::  Your Task

   - Use a parameter for the subscriber's topic
   - Update the car's position as if it was driving with the speed that is received from the subscriber. E.g. if you publish the "speed: 5.0" message, the car should drive with 5 m/s.

After completing the task, don't forget to commit and push your results!

