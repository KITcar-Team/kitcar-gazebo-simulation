.. _models:

Models and Sensors
==================

Gazebo `model and sensor plugins <http://gazebosim.org/tutorials?tut=ros_gzplugins>`_ make \
for a great ROS integration.
Model plugins enable interacting with models inside the simulated scene and sensor plugins \
can publish the output of sensors as ROS messages.

Dr. Drift
---------

Dr. Drift's model definition is a good place to understand how \
these two elements are utilized to make the simulation work.
The complete definition of Dr. Drift in ``simulation/models/model_db/dr_drift/model.sdf`` \
is quite long and contains multiple cameras. However, only a view lines of the definition \
are sufficient to grasp what's going on.

.. warning::

   The following code snippets are not included from the actual model.sdf \
   and details might differ.
   Nevertheless, the principle ideas stay the same.

The line

.. code-block:: xml

   <plugin filename="libmodel_plugin_link.so" name="model_plugin_link"/>

includes the model plugin **model_plugin_link**.
With it, the car's pose and twist can be controlled through ROS topics.
(See :ref:`gazebo_simulation` for more.)

The lines

.. code-block:: xml

            <sensor name="camera" type="camera">
                <update_rate>60</update_rate>
                <camera name="__default__">
                    <horizontal_fov>2.076915850284772</horizontal_fov>
                    <image>
                        <width>1280</width>
                        <height>1024</height>
                        <format>R8G8B8</format>
                    </image>
                    <clip>
                        <near>0.1</near>
                        <far>2.5</far>
                    </clip>
                </camera>
                <plugin filename="libgazebo_ros_camera.so" name="camera_plugin">
                    <alwaysOn>1</alwaysOn>
                    <updateRate>0</updateRate>
                    <imageTopicName>/simulation/sensors/raw/camera</imageTopicName>
                    <cameraInfoTopicName>/simulation/sensors/info/camera</cameraInfoTopicName>
                    <frameName>camera</frameName>
                </plugin>
            </sensor>

define the car's front camera.
Besides the actual sensor properties, a sensor plugin called "camera_plugin" is added.
The sensor plugin publishes the camera image on the ROS topic \
``"/simulation/sensors/raw/camera"`` on which it is then available to other nodes.
