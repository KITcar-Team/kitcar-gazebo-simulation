Simulation CI Image
-------------------

This image is intended to be used as a base image for the Gitlab-CI.
It is not supposed to work out of the box,
just as a starting point to speed up the CI pipeline.

The image can be built by running the **build.sh** script:

.. prompt:: bash

  ./build.sh ${CI_REGISTRY} ${IMAGE_TAG}

from ``simulation/docker/ci``.
