import argparse
import os
import pickle

import torch

import simulation.utils.machine_learning.data as ml_data
from simulation.utils.machine_learning.cycle_gan.configs.test_options import (
    WassersteinCycleGANTestOptions,
    CycleGANTestOptions,
)
from simulation.utils.machine_learning.cycle_gan.models.cycle_gan_model import CycleGANModel
from simulation.utils.machine_learning.cycle_gan.models.generator import create_generator
from simulation.utils.machine_learning.cycle_gan.models.wcycle_gan import (
    WassersteinCycleGANModel,
)
from simulation.utils.machine_learning.data.image_operations import save_images
from simulation.utils.machine_learning.models.helper import get_norm_layer, init_net
from simulation.utils.machine_learning.models.resnet_generator import ResnetGenerator

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
        netg_a = ResnetGenerator(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            get_norm_layer(opt.norm),
            dilations=opt.dilations,
            conv_layers_in_block=opt.conv_layers_in_block,
        )
    else:
        netg_a = create_generator(
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

    netg_b = pickle.loads(pickle.dumps(netg_a))

    netg_a = init_net(netg_a, opt.init_type, opt.init_gain, device)
    netg_b = init_net(netg_b, opt.init_type, opt.init_gain, device)

    ModelClass = CycleGANModel if not opt.is_wgan else WassersteinCycleGANModel

    model = ModelClass.from_options(netg_a=netg_a, netg_b=netg_b, **opt.to_dict())

    model.networks.load(
        os.path.join(opt.checkpoints_dir, opt.name, f"{opt.epoch}_net_"),
        device=device,
    )
    model.networks.print(opt.verbose)

    model.eval()
    for i, ((a, _), (b, _)) in enumerate(zip(dataset_a, dataset_b)):
        a = a.to(device)
        b = b.to(device)
        stats = model.test(a, b)  # run inference
        visuals = stats.get_visuals()
        if i % 5 == 0:
            print("processing (%04d)-th image." % i)
        save_images(
            visuals=visuals,
            destination=os.path.join(opt.results_dir, opt.name),
            aspect_ratio=opt.aspect_ratio,
            post_fix=str(i),
        )
