ROS Parameters
==============

In the previous part of the Onboarding, you made the car drive on the road \
using a publisher and :py:func:`steer`.

However, you still have many hard-coded constants within your code.
E.g. if you want to change the update rate with which :py:func:`steer` \
gets called or change the publisher's topic, you will need to modify your
source code.
This makes your code less flexible and harder to understand.
As you've seen in :ref:`onboarding_ros_package`, ROS can load parameters from a parameter file; \
making them accessible in your node.
When subclassing from :py:class:`simulation.utils.ros_base.NodeBase`,
you can access the parameters inside the node with

>>> self.param.{NAME_OF_PARAMETER}  # doctest: +SKIP

Our nodes usually have two parameter files:

- *default.yaml*: Define general parameters
- *topics.yaml*: Define topics used by subscribers and publishers

Parameter files are yaml files.
See https://learn.getgrav.org/16/advanced/yaml for a quick introduction \
into the yaml-syntax.

The parameter files are located in ``path/to/ros_package/param/<NAME_OF_NODE>``.

Let's take a look at the :ref:`onboarding_node`'s parameter files \
and how they are included when launching the node.

The *default.yaml* currently only defines two parameters; *param_name_1* and *param_name_2*:

.. literalinclude:: ../../../simulation/src/simulation_onboarding/param/onboarding/default.yaml
   :language: yaml

The *topics.yaml* also defines one topic:

.. literalinclude:: ../../../simulation/src/simulation_onboarding/param/onboarding/topics.yaml
   :language: yaml

After defining the parameters, they must also be included in the launch file.
In the node's launch file, the parameters are loaded using **<rosparam>**:

.. literalinclude:: ../../../simulation/src/simulation_onboarding/launch/onboarding_node.launch
   :language: xml

Inside the node, you can now access the parameters with:

.. doctest::
   :options: +SKIP

   >>> print(self.param.param_name_1)
   'param_value_1'
   >>> print(self.param.param_name_2)
   'param_value_2'

When taking a closer look, you can see that topics are loaded with *ns="topics"*.
Whenever

.. code-block:: xml

  <rosparam ... ns="<NAMESPACE>" />

is provided, parameters from that file are accessible through:

>>> self.param.NAMESPACE...  # doctest: +SKIP

Therefore, the topic can be accessed with:

>>> print(self.param.topics.topic_name)  # doctest: +SKIP
'topic'

Replace Constants with Parameters
---------------------------------

.. admonition:: Your Task

   - Create and use a *rate*-parameter for the update frequency of the steer function.
   - Add a topic parameter *set_pose* for the publisher's topic.

After completing the task, don't forget to commit and push your results!
