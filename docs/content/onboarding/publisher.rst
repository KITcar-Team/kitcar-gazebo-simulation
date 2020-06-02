ROS Publisher
=============

In this part of the Onboarding, you will let the car drive on the onboarding road.
You will do so by creating a publisher inside :py:class:`OnboardingNode`.
The publisher will be able to send messages that modify the car's position.

Gazebo Plugins
--------------

Before getting started, we need to take a closer look at Gazebo.
You already know how you can start a road in Gazebo and move the car around by hand.
Gazebo also allows to programmatically change the position (and other properties) of models.
In particular, plugins can be defined and attached to multiple models, sensors, and more,
to modify a wide variety of properties.

If you take a look **Dr.Drift**'s model definition in \
``simulation/src/gazebo_simulation/param/car_specs/dr_drift/model.urdf``
you will find the line

.. code-block:: xml

   <plugin name="model_plugin_link" filename="libmodel_plugin_link.so"/>


which equipes the *model_plugin_link*.

The *model_plugin_link* is defined in :ref:`gazebo_simulation`.
When attached to a model it listens to topics through which the pose and \
twist of the model can be modified.
In this case, we only want to modify the car's pose. The *model_plugin_link* 's topic \
for setting a pose is:

.. code-block:: shell

   /simulation/gazebo/model/NAME_OF_MODEL/set_pose

Since the name of the car is *dr_drift*, our topic of interest is */simulation/gazebo/model/dr_drift/set_pose*.

The new pose has to be sent as a
SetModelPose message.
It is defined in ``simulation/src/gazebo_simulation/msg/SetModelPose.msg``:

.. literalinclude:: ../../../simulation/src/gazebo_simulation/msg/SetModelPose.msg

Create the Publisher
--------------------

To control the car you will first need to add a publisher with topic */simulation/gazebo/model/dr_drift/set_pose*
and message type *SetModelPose* to the :py:class:`OnboardingNode`:

1) Import ROS and *SetModelPose* at the top of the file:

.. code-block:: python

   import rospy
   from gazebo_simulation.msg import SetModelPose

2) Create the publisher in :py:func:`start`:

.. code-block:: python

   self.pose_publisher = rospy.Publisher(
       name="/simulation/gazebo/model/dr_drift/set_pose",
       data_class=SetModelPose,
       queue_size=1,
   )

See http://docs.ros.org/lunar/api/rospy/html/rospy.topics.Publisher-class.html for a detailed explanation.


3) Unregister the publisher in stop():

.. code-block:: python

   self.pose_publisher.unregister()


Publish messages
----------------

The following lines show you how the publisher can be used to set the car's x-coordinate *x=0*:

.. code-block:: python

   pose_msg = SetModelPose()  # Create pose msg

   # Append x coordinate to keys and values
   pose_msg.keys.append(SetModelPose.POSITION_X)
   pose_msg.values.append(0)

   self.pose_publisher.publish(pose_msg)


Add these lines to :py:func:`steer` to test if setting the x-coordinate works. Start Gazebo and the node with roslaunch.

.. prompt:: bash

   roslaunch simulation_onboarding master.launch road:=onboarding_simple

When you try to move the car around it should always be set back onto the x-axis.


Drive along the road
--------------------

You now know how to control the car's x-coordinate.

.. admonition:: Your Task

   Make the car drive on the right side of the road with *1* m/s.
   Once it is at the end of the road, put it back to the beginning of the road!

After completing the task, don't forget to commit and push your results!

.. hint::

   Use :py:func:`rospy.Time.now().to_sec()` to get the current time in seconds.
   The road is 5.8 meters long.
