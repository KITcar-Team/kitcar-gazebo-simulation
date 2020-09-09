import os
import pathlib

import cv2
import numpy as np
import torch
from PIL import Image

from simulation.utils.machine_learning.data.base_dataset import get_transform
from .configs.test_options import CycleGANTestOptions, WassersteinCycleGANTestOptions
from .models.cycle_gan_model import CycleGANModel
from ..data.image_operations import tensor2im


class RosConnector:
    """Implementation of a simple ROS interface to translate simulated to "real" images."""

    def __init__(self, use_wasserstein=True):
        """Initialize the RosConnector class

        Use default test options but could be via command-line. Load and setup the model
        """

        opt = WassersteinCycleGANTestOptions if use_wasserstein else CycleGANTestOptions

        opt.checkpoints_dir = os.path.join(
            pathlib.Path(__file__).parent.absolute(), opt.checkpoints_dir
        )

        self.model = CycleGANModel.from_options(
            **opt.to_dict()
        )  # create a model given model and other options
        self.model.setup(
            verbose=opt.verbose,
            load_iter=opt.load_iter,
        )
        # self.model.eval()
        tf_properties = {
            "load_size": opt.load_size,
            "crop_size": opt.crop_size,
            "preprocess": opt.preprocess,
            "mask": os.path.join(pathlib.Path(__file__).parent.absolute(), opt.mask),
            "no_flip": True,
            "grayscale": True,
        }
        self.transform = get_transform(**tf_properties)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Translate an image to a "fake real" image by using the loaded model.

        Args:
            image: Image to be translated to "fake real"

        Returns:
            Translated image.
        """
        # Store shape
        h, w = image.shape

        # Convert to PIL
        image = Image.fromarray(image)

        # Apply transformations
        image: torch.Tensor = self.transform(image)

        # Copy the numpy array because it's not writeable otherwise
        # Bring into shape [1,1,h,w]
        image.unsqueeze_(0)

        # Inference
        result = self.model.netg_a.forward(image).detach()

        # From [-1,1] to [0,256]
        result = tensor2im(result, to_rgb=False)

        # Resize to the size the input image has
        result = cv2.resize(result, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        # Return as mono8 encoding
        return result
