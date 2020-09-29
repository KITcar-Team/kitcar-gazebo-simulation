import argparse
import os
import pickle

import torch
from torch.nn.functional import mse_loss

import simulation.utils.machine_learning.data as ml_data
from simulation.utils.machine_learning.cycle_gan.configs.test_options import (
    CycleGANTestOptions,
    WassersteinCycleGANTestOptions,
)
from simulation.utils.machine_learning.cycle_gan.models.cycle_gan_model import CycleGANModel
from simulation.utils.machine_learning.cycle_gan.models.discriminator import (
    create_discriminator,
)
from simulation.utils.machine_learning.cycle_gan.models.generator import create_generator
from simulation.utils.machine_learning.cycle_gan.models.wcycle_gan import (
    WassersteinCycleGANModel,
)
from simulation.utils.machine_learning.models.helper import get_norm_layer, init_net
from simulation.utils.machine_learning.models.resnet_generator import ResnetGenerator
from simulation.utils.machine_learning.models.wasserstein_critic import WassersteinCritic


def calculate_loss_d(
    discriminator: torch.nn.Module, real: torch.Tensor, fake: torch.Tensor
) -> float:
    def gan_loss(prediction: torch.Tensor, is_real: bool):
        target = torch.tensor(1.0 if is_real else 0.0, device=prediction.device).expand_as(
            prediction
        )
        return mse_loss(prediction, target)

    pred_real = discriminator(real)
    loss_d_real = gan_loss(pred_real, True)
    pred_fake = discriminator(fake)
    loss_d_fake = gan_loss(pred_fake, False)
    loss_d = (loss_d_real + loss_d_fake) * 0.5
    return loss_d.item()


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

    dataset_size = min(
        len(dataset_a), len(dataset_b)
    )  # get the number of images in the dataset.

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
        netd_a = WassersteinCritic(
            opt.input_nc,
            ndf=opt.ndf,
            dilations=opt.dilations,
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
        netd_a = create_discriminator(
            opt.input_nc, opt.ndf, opt.netd, opt.n_layers_d, opt.norm, opt.use_sigmoid
        )

    netg_b = pickle.loads(pickle.dumps(netg_a))
    netd_b = pickle.loads(pickle.dumps(netd_a))

    netg_a = init_net(netg_a, opt.init_type, opt.init_gain, device)
    netg_b = init_net(netg_b, opt.init_type, opt.init_gain, device)
    netd_a = init_net(netd_a, opt.init_type, opt.init_gain, device)
    netd_b = init_net(netd_b, opt.init_type, opt.init_gain, device)

    ModelClass = CycleGANModel if not opt.is_wgan else WassersteinCycleGANModel

    model = ModelClass.from_dict(
        netg_a=netg_a, netg_b=netg_b, netd_a=netd_a, netd_b=netd_b, **opt.to_dict()
    )

    model.networks.load(
        os.path.join(opt.checkpoints_dir, opt.name, f"{opt.epoch}_net_"),
        device=device,
    )
    model.networks.print(opt.verbose)
    model.eval()

    sum_loss_a = 0
    sum_loss_b = 0

    for i, ((batch_a, _), (batch_b, _)) in enumerate(zip(dataset_a, dataset_b)):
        batch_a = batch_a.to(device)
        batch_b = batch_b.to(device)

        real_a = batch_a
        fake_b = model.networks.g_a(real_a)
        real_b = batch_b
        fake_a = model.networks.g_b(real_b)

        sum_loss_a += calculate_loss_d(model.networks.d_a, real_b, fake_b)
        sum_loss_b += calculate_loss_d(model.networks.d_b, real_a, fake_a)

        print(f"Processing {100 * i/dataset_size:.2f}%")

    print(f"AVG-Loss Discriminator A: {sum_loss_a / dataset_size}")
    print(f"AVG-Loss Discriminator B: {sum_loss_b / dataset_size}")

    file_path = os.path.join(opt.results_dir, opt.name, "discriminator_losses.txt")
    with open(file_path, "w") as file:
        file.write(f"{sum_loss_a / dataset_size},{sum_loss_b / dataset_size}")
