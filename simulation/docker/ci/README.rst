Simulation CI Image
-------------------

This image is intended to be used as a base image for the Gitlab-CI.
It is not supposed to work out of the box,
just as a starting point to speed up the CI pipeline.

The image can be built by running the **build.sh** script:

.. prompt:: bash

  ./build.sh ${SERVICE} ${IMAGE_TAG} ${PARENT_TAG}

from ``simulation/docker/ci``.

Dependending on the provided `${SERVICE}` different packages are installed:

* `${SERVICE}=base`: No additional packages.
* `${SERVICE}=machine_learning`: Additional machine learning pip packages
  and NodeJS are installed. Both are necessary to train or test neural networks and
  publish the results using CML.
* `${SERVICE}=docs`: Additional machine learning and documentation pip packages
  are installed. Both are necessary to build Sphinx.

The services **docs** and **machine_learning** inherit from **base**.
For that purposes the parent's image must be passed when building **docs** or **machine_learning**.

The dependency graph is: **base** -> **machine_learning** -> **docs**.
(Because the machine learning packages are required when building the documentation. However, the
NodeJS is not actually required, but is installed to simplify the dependencies.)
