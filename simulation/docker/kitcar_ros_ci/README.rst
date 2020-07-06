Gazebo Kitcar ROS Image
-----------------------

This image is intended to be used within the kitcar-ros CI pipeline.

The image can be built by running the **build.sh** script:

.. prompt:: bash

   ./build.sh ${CI_REGISTRY} ${PARENT_TAG} ${IMAGE_TAG}

from ``simulation/docker/kitcar_ros_ci``.
