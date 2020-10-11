.. _datasets:

Datasets
========

Developing data hungry code, e.g. machine learning applications, comes with the burden of
creating and maintaining datasets.
While creating a dataset might seem to be most of the work, often enough, maintaining it is
the hard part.

.. admonition:: Example: Cycle GAN

   The cycle GAN requires a set of real images captured on the real car and a set of
   simulated images recorded within the simulation.
   After training, the resulting generator is able to translate simulated images into
   real looking images.

   However, everytime something is changed within the simulation; even something small
   in the background, the cycle GAN must be retrained with up to date images from the
   simulation.

DVC is used in combination with scripts explained in :ref:`rosbags` to dramatically
reduce the effort spent keeping the datasets up to date.

The principal idea is to use dvc pipelines, i.e. multiple commands with meta-information
about dependencies and outputs that are defined in *dvc.yaml* files, to reproducibly create
datasets that solely depend on the simulation.
Whenever the dependencies change, the pipeline can be rerun and the updated dataset
generated.


Automatic Labeling
-------------------

A key benefit of simulations is that all information about the world is available.
This fact can be used automatically generate labeled data using the simulation.

First, a rosbag of the simulation running with the :ref:`label_camera_node` is recorded
and in a second step, the necessary information extracted from the rosbag using scripts from
:ref:`rosbags`.
The result is a folder of images with an additional yaml file that contains all labels.
With :py:mod:`simulation.utils.machine_learning.data.labeled_dataset` the folder can be
used as a dataset.
