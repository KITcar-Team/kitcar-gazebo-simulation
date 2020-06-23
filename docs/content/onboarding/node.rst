.. _onboarding_node:

########
ROS Node
########

In the previous part, we already mentioned the node. In this chapter, we will get into the
details.

Let's start by taking a look at the complete file again:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/src/onboarding/node.py
   :language: python
   :linenos:

The :py:class:`OnboardingNode` class is a subclass of
:mod:`NodeBase<simulation.utils.ros_base.node_base>` (Go check it out). This speeds up the
development process by handling all rospy related functions.

********
__init__
********

Let's start by looking at :py:func:`__init__`:

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/src/onboarding/node.py
   :language: python
   :linenos:
   :start-after: ROS node to teach new members.
   :end-before: def start

The function :py:func:`super` returns the next super class of :py:class:`self`.
(You can take a look at one of our short talks for more: :ref:`talk_inheritance`.)
In this case, it's :py:class:`NodeBase`.
The method :py:func:`__init__` gets called with the argument *name*.
This introduces the node to ROS and loads parameters specified inside
``onboarding_node.launch``.

Next, :py:func:`self.run` is called.
It is defined within :py:class:`NodeBase` and will correctly start \
and shutdown the node by calling :py:func:`self.start` and :py:func:`self.stop`.
When passing *function=*:py:class:`self.steer` and *rate=60* as arguments,
:py:class:`NodeBase` will also call the function :py:func:`self.steer` 60 times a second.


*****
start
*****

:py:func:`start` is called before when the node is started.
In this example, nothing happens, but the supers' :py:func:`start` method gets called.
At the moment it doesn't do a thing but you will need to add code here later.

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/src/onboarding/node.py
   :language: python
   :linenos:
   :start-after: self.run(
   :end-before: def stop

****
stop
****

:py:func:`stop` is much like *start*.
It is called when the node is shutting down.
And again it doesn't do anything at the moment.

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/src/onboarding/node.py
   :language: python
   :linenos:
   :start-after: super().start(
   :end-before: def steer

*****
steer
*****

As described earlier, :py:func:`steer` is called 60 times a second.
It is empty, but you will add code to it later in this tutorial.

.. literalinclude::
   ../../../simulation/src/simulation_onboarding/src/onboarding/node.py
   :language: python
   :linenos:
   :start-after: super().stop(
