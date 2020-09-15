import argparse
import os
import pickle
from typing import Tuple

import torch
from torch import nn

import simulation.utils.machine_learning.data as ml_data
from simulation.utils.machine_learning.cycle_gan.configs.test_options import (
    CycleGANTestOptions,
    WassersteinCycleGANTestOptions,
)
from simulation.utils.machine_learning.cycle_gan.models.cycle_gan_model import CycleGANModel
from simulation.utils.machine_learning.cycle_gan.models.generator import create_generator
from simulation.utils.machine_learning.cycle_gan.models.wcycle_gan import (
    WassersteinCycleGANModel,
)
from simulation.utils.machine_learning.data import UnlabeledDataLoader
from simulation.utils.machine_learning.data.image_operations import save_images
from simulation.utils.machine_learning.models.helper import get_norm_layer, init_net
from simulation.utils.machine_learning.models.resnet_generator import ResnetGenerator


def test_on_dataset(
    dataset: UnlabeledDataLoader,
    generators: Tuple[nn.Module, nn.Module],
    class_names: Tuple[str, str],
    destination: str,
    aspect_ratio: float = 1,
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    """Test one dataset and save all images.

    Args:
        dataset: The dataset to test
        generators: Both generators. Second one is used to generate the fake image.
        class_names: The class names to save the images correctly.
        destination: The destination folder
        aspect_ratio: The aspect ratio of the images
        device: the device on which the models are located
    """
    for i, (real_image, _) in enumerate(dataset):
        real_image = real_image.to(device)
        fake_image = generators[0](real_image)
        idt_image = generators[1](real_image)
        cycle_image = generators[1](fake_image)
        visuals = {
            f"real_{class_names[0]}": real_image,
            f"fake_{class_names[1]}": fake_image,
            f"idt_{class_names[0]}": idt_image,
            f"cycle_{class_names[0]}": cycle_image,
        }
        print(f"Processing {i}-th image on dataset {class_names[0]}.")
        save_images(
            visuals=visuals,
            destination=destination,
            aspect_ratio=aspect_ratio,
            post_fix=str(i),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gan_type",
        type=str,
        default="default",
        help="Decide whether to use Wasserstein gan or default gan [default, wgan]",
    )
    use_wasserstein = parser.parse_args().gan_type == "wgan"

    opt = WassersteinCycleGANTestOptions if use_wasserstein else CycleGANTestOptions

    tf_properties = {
        "load_size": opt.load_size,
        "crop_size": opt.crop_size,
        "preprocess": opt.preprocess,
        "mask": opt.mask,
    }
    dataset_a, dataset_b = ml_data.load_unpaired_unlabeled_datasets(
        opt.dataset_a,
        opt.dataset_b,
        batch_size=1,
        sequential=True,
        num_threads=0,
        grayscale_a=(opt.input_nc == 1),
        grayscale_b=(opt.output_nc == 1),
        max_dataset_size=opt.max_dataset_size,
        transform_properties=tf_properties,
    )  # create datasets for each domain (A and B)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt.is_wgan:
        netg_a_to_b = ResnetGenerator(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            get_norm_layer(opt.norm),
            dilations=opt.dilations,
            conv_layers_in_block=opt.conv_layers_in_block,
        )
    else:
        netg_a_to_b = create_generator(
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

    netg_b_to_a = pickle.loads(pickle.dumps(netg_a_to_b))

    netg_a_to_b = init_net(netg_a_to_b, opt.init_type, opt.init_gain, device)
    netg_b_to_a = init_net(netg_b_to_a, opt.init_type, opt.init_gain, device)

    ModelClass = CycleGANModel if not opt.is_wgan else WassersteinCycleGANModel

    model = ModelClass.from_dict(
        netg_a_to_b=netg_a_to_b, netg_b_to_a=netg_b_to_a, **opt.to_dict()
    )

    model.networks.load(
        os.path.join(opt.checkpoints_dir, opt.name, f"{opt.epoch}_net_"),
        device=device,
    )
    model.networks.print(opt.verbose)
    model.eval()

    test_on_dataset(
        dataset_a,
        (model.networks.g_a_to_b, model.networks.g_b_to_a),
        ("a", "b"),
        destination=os.path.join(opt.results_dir, opt.name),
        aspect_ratio=opt.aspect_ratio,
        device=device,
    )
    test_on_dataset(
        dataset_b,
        (model.networks.g_b_to_a, model.networks.g_a_to_b),
        ("b", "a"),
        destination=os.path.join(opt.results_dir, opt.name),
        aspect_ratio=opt.aspect_ratio,
        device=device,
    )
