Docker Images
-------------

The images can be built by running **docker-compose**:

.. prompt:: bash

  docker-compose ${SERVICE}

from ``docker/``.

Depending on the provided `${SERVICE}` different packages are installed:

* `${SERVICE}=base`: No additional packages.
* `${SERVICE}=cml`: Pytorch with CUDA and the CML-Bot are installed.
  Both are necessary to train or test neural networks and publish the results using CML.
* `${SERVICE}=kitcar-ros`: This image is intended to be used within the kitcar-ros CI pipeline.
  It adds the whole simulation code and installs some more packages,
  which are necessary for running the simulation within the kitcar ros CI.


The services **cml** and **kitcar-ros** inherit from **base**.
