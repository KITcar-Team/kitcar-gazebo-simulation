Docker Images
=============

There are multiple Docker images defined within ``simulation/docker``.
They serve as images for our Gitlab CI or may be used for other development purposes.
Generally, there is a **build.sh** script for every image that allows to generate the image without further work necessary.

The images are also **built** within the Gitlab CI pipeline and **pushed** to our Gitlab Docker registry.
After logging into the registry

.. prompt:: bash

   docker login git.kitcar-team.de:4567

all images available can be pulled.

An image built locally can be uploaded to the Gitlab Docker registry with:

.. prompt:: bash

   docker push git.kitcar-team.de:4567/kitcar/kitcar-gazebo-simulation/...


.. include:: ../../../simulation/docker/ci/README.rst

.. include:: ../../../simulation/docker/default/README.rst

.. include:: ../../../simulation/docker/kitcar_ros_ci/README.rst
