"""The Cycle GAN can be used to convert simulated images into real looking images.

During training, a class A image is "translated" to class B using a generator
and "retranslated" to class A using another generator.
The difference between these images is the error of the two networks,
which should be minimized by the training process.
This process also goes from class B to A and back to B.
The discriminators are used to evaluate whether an image originates from
the class A or class B. They are trained for this classification at the same time.

PyTorch is used for the implementation.

Dataset
-------

The dataset consists of 2 classes of images.

    - real images (Class A)
    - simulated images (Class B)

`DVC <https://dvc.org/>`_ is used for dataset management.
With DVC even large datasets can be versioned quickly and easily.
The data sets are located under data/real_images and data/simulated_images.
With dvc pull the data sets are downloaded:

.. prompt:: bash

   dvc pull

Also trained models are stored with DVC.
A new model can be added by running with dvc commit.

.. prompt:: bash

   dvc commit
   dvc push

.. important::
    After a git checkout there should always be a dvc checkout,
    otherwise the data can get mixed up.

    .. prompt:: bash

       dvc checkout

*This is only a short summary of how dvc works, a better explanation can be found here:*
`DVC Tutorial <https://dvc.org/doc/start>`_

Training
--------

All parameters for training can be found in
``simulation/utils/machine_learning/config.yaml``.

A new model can be trained with the training script.

.. prompt:: bash

   python3 train.py

The training script automatically starts a visdom server,
which shows the current state of the training.
The script automatically saves the model in between,
these intermediate states are stored in the checkpoints folder.

Testing
-------

For testing, checkpoints must already exist.

All parameters for testing can be found in
``simulation/utils/machine_learning/config.yaml``.
There are two ways to test a model.

**With DVC stages**

For testing purposes there are the following DVC stages:

    - test_dr_drift_256
    - make_video_dr_drift_256

A DVC stage is executed with the following command:

.. prompt:: bash

   dvc repro -s NAME_OF_STAGE

**test_dr_drift_256**:
Tests the model by loading the last checkpoints from the folder
checkpoints/dr_drift_256 and creates the results folder with the test results

**make_video_dr_drift_256**:
Takes the results from the results folder and cuts 3 videos together.
One with the pictures from the simulation,
another one with the "translated" images
and one where the two videos were stacked on top of each other others.

**stacked_gif_dr_drift_256**:
Converts the stacked video into a short gif.

**Without DVC stages**

A model can be tested with the testing script.

.. prompt:: bash

   python3 test.py

The test script creates a new folder with all generated images in it.
The resulting images can be cut into a video with the script
``simulation/utils/machine_learning/data/images_to_video.py``.

With CML in the pipeline, each new model is automatically tested
and the results are presented in a report under the commit or merge request.
This requires that the model be named "dr_drift_256".
"""
