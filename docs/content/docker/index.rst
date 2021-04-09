Docker Images
=============

There are multiple Docker images defined within ``docker/``.
They serve as images for our Gitlab CI or may be used for other development purposes.

The images are also **built** within the Gitlab CI pipeline and **pushed** to our Gitlab Docker registry.
After logging into the registry

.. prompt:: bash

   docker login git.kitcar-team.de:4567

all images available can be pulled.

An image built locally can be uploaded to the Gitlab Docker registry with:

.. prompt:: bash

   docker push git.kitcar-team.de:4567/kitcar/kitcar-gazebo-simulation/...

.. include:: ../../../docker/README.rst

Docker Compose
--------------

For simplification we have introduced a docker compose file.
It contains the main information about the build configuration of our docker images.

.. literalinclude:: ../../../docker/docker-compose.yaml
   :language: yaml
