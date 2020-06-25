Gazebo CI Docker Image
======================

This image is intended to be used as a base image for the Gitlab-CI.
It is not supposed to work out of the box,
just as a starting point to speed up the CI pipeline.

Usually, the image is built and updated automatically within the Gitlab CI.

Build
-----

The image can be built by running the **build.sh** script::

  ./build.sh

from this directory.

Push
----

The image should then be pushed to Gitlab.
You can do so by logging into the registry::

  docker login git.kitcar-team.de

And then upload the image to the Gitlab Docker registry::

  docker push git.kitcar-team.de:4567/kitcar/kitcar-gazebo-simulation:focal

Once that is done, the image will be used by the CI pipeline.
