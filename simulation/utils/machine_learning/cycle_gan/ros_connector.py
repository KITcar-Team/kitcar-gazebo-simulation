import argparse
import os
import pathlib

import cv2
import numpy as np
import torch
from PIL import Image

from simulation.utils.machine_learning.data.base_dataset import get_transform
from simulation.utils.machine_learning.data.image_operations import tensor2im
from simulation.utils.machine_learning.models import resnet_generator
from simulation.utils.machine_learning.models.helper import get_norm_layer, init_net

from .configs.test_options import CycleGANTestOptions, WassersteinCycleGANTestOptions
from .models import generator
from .models.cycle_gan_model import CycleGANModel
from .models.wcycle_gan import WassersteinCycleGANModel


class RosConnector:
    """Implementation of a simple ROS interface to translate simulated to "real" images."""

    def __init__(self, use_wasserstein=True):
        """Initialize the RosConnector class.

        Use default test options but could be via command-line. Load and setup the model
        """

        opt = WassersteinCycleGANTestOptions if use_wasserstein else CycleGANTestOptions

        opt.checkpoints_dir = os.path.join(
            pathlib.Path(__file__).parent.absolute(), opt.checkpoints_dir
        )

        tf_properties = {
            "load_size": opt.load_size,
            "crop_size": opt.crop_size,
            "preprocess": opt.preprocess,
            "mask": os.path.join(os.path.dirname(__file__), opt.mask),
            "no_flip": True,
            "grayscale": True,
        }
        self.transform = get_transform(**tf_properties)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if opt.is_wgan:
            netg_b_to_a = resnet_generator.ResnetGenerator(
                opt.input_nc,
                opt.output_nc,
                opt.ngf,
                get_norm_layer(opt.norm),
                dilations=opt.dilations,
                conv_layers_in_block=opt.conv_layers_in_block,
            )
        else:
            netg_b_to_a = generator.create_generator(
                opt.input_nc,
                opt.output_nc,
                opt.ngf,
                opt.netg,
                opt.norm,
                not opt.no_dropout,
                opt.activation,
                opt.conv_layers_in_block,
                opt.dilations,
            )

        netg_b_to_a = init_net(netg_b_to_a, opt.init_type, opt.init_gain, self.device)

        ModelClass = CycleGANModel if not opt.is_wgan else WassersteinCycleGANModel

        self.model = ModelClass.from_dict(
            netg_a_to_b=None, netg_b_to_a=netg_b_to_a, **opt.to_dict()
        )

        self.model.networks.load(
            os.path.join(opt.checkpoints_dir, opt.name, f"{opt.epoch}_net_"),
            device=self.device,
        )
        self.model.eval()

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
        image = image.to(self.device)

        # Copy the numpy array because it's not writeable otherwise
        # Bring into shape [1,1,h,w]
        image.unsqueeze_(0)

        # Inference
        result = self.model.networks.g_b_to_a.forward(image).detach()

        # From [-1,1] to [0,256]
        result = tensor2im(result, to_rgb=False)

        # Resize to the size the input image has
        result = cv2.resize(result, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        # Return as mono8 encoding
        return result


if __name__ == "__main__":
    """Run GAN over all files in folder."""
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--input_dir", help="Directory with input images.")
    parser.add_argument("--output_dir", help="Directory for output images.")
    parser.add_argument(
        "--gan_type",
        type=str,
        default="default",
        help="Decide whether to use Wasserstein gan or default gan [default, wgan]",
    )
    args = parser.parse_args()
    GAN = RosConnector(args.gan_type)

    files = [
        file
        for file in os.listdir(args.input_dir)
        if os.path.isfile(os.path.join(args.input_dir, file))
        and file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ]
    os.makedirs(args.output_dir, exist_ok=True)

    for i, file in enumerate(files):
        input_file_path = os.path.join(args.input_dir, file)
        output_file_path = os.path.join(args.output_dir, file)

        translated_image = GAN(np.array(Image.open(input_file_path)))
        cv2.imwrite(output_file_path, translated_image)

        print(f"Processing: {100 * i / len(files):.2f}%")
