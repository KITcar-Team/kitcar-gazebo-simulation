import argparse
import os
import pickle
import time
from contextlib import suppress

import torch
from torch.nn.modules.module import ModuleAttributeError

import simulation.utils.machine_learning.data as ml_data
from simulation.utils.machine_learning.cycle_gan.configs.train_options import (
    CycleGANTrainOptions,
    WassersteinCycleGANTrainOptions,
)
from simulation.utils.machine_learning.cycle_gan.models.cycle_gan_model import CycleGANModel
from simulation.utils.machine_learning.cycle_gan.models.discriminator import (
    create_discriminator,
)
from simulation.utils.machine_learning.cycle_gan.models.generator import create_generator
from simulation.utils.machine_learning.cycle_gan.models.wcycle_gan import (
    WassersteinCycleGANModel,
)
from simulation.utils.machine_learning.data.visualizer import Visualizer
from simulation.utils.machine_learning.models.helper import get_norm_layer, init_net
from simulation.utils.machine_learning.models.resnet_generator import ResnetGenerator
from simulation.utils.machine_learning.models.wasserstein_critic import WassersteinCritic

if __name__ == "__main__":  # noqa C901
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gan_type",
        type=str,
        default="default",
        help="Decide whether to use Wasserstein gan or default gan [default, wgan]",
    )
    use_wasserstein = parser.parse_args().gan_type == "wgan"

    opt = WassersteinCycleGANTrainOptions if use_wasserstein else CycleGANTrainOptions

    tf_properties = {
        "load_size": opt.load_size,
        "crop_size": opt.crop_size,
        "preprocess": opt.preprocess,
        "mask": opt.mask,
        "no_flip": opt.no_flip,
    }
    dataset_a, dataset_b = ml_data.load_unpaired_unlabeled_datasets(
        opt.dataset_a,
        opt.dataset_b,
        batch_size=opt.batch_size,
        sequential=False,
        num_threads=opt.num_threads,
        grayscale_a=(opt.input_nc == 1),
        grayscale_b=(opt.output_nc == 1),
        max_dataset_size=opt.max_dataset_size,
        transform_properties=tf_properties,
    )  # create datasets for each domain (A and B)

    dataset_size = max(
        len(dataset_a), len(dataset_b)
    )  # get the number of images in the dataset.
    print("The number of training images = %d" % dataset_size)

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
        netd_a = WassersteinCritic(
            opt.input_nc,
            ndf=opt.ndf,
            dilations=opt.dilations,
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
        netd_a = create_discriminator(
            opt.input_nc,
            opt.ndf,
            opt.netd,
            opt.n_layers_d,
            opt.norm,
            opt.use_sigmoid,
        )

    netg_b_to_a = pickle.loads(pickle.dumps(netg_a_to_b))
    netd_b = pickle.loads(pickle.dumps(netd_a))

    netg_a_to_b = init_net(netg_a_to_b, opt.init_type, opt.init_gain, device)
    netg_b_to_a = init_net(netg_b_to_a, opt.init_type, opt.init_gain, device)
    netd_a = init_net(netd_a, opt.init_type, opt.init_gain, device)
    netd_b = init_net(netd_b, opt.init_type, opt.init_gain, device)

    ModelClass = CycleGANModel if not opt.is_wgan else WassersteinCycleGANModel

    model = ModelClass.from_dict(
        netg_a_to_b=netg_a_to_b,
        netg_b_to_a=netg_b_to_a,
        netd_a=netd_a,
        netd_b=netd_b,
        **opt.to_dict(),
    )

    model.create_schedulers(
        lr_policy=opt.lr_policy,
        lr_decay_iters=opt.lr_decay_iters,
        lr_step_factor=opt.lr_step_factor,
        n_epochs=opt.n_epochs,
    )  # regular setup: load and print networks; create schedulers

    if opt.continue_train:
        try:
            model.networks.load(
                os.path.join(opt.checkpoints_dir, opt.name, f"{opt.epoch}_net_"),
                device=device,
            )
        except FileNotFoundError:
            print("Could not load model weights. Proceeding with new weights.")
        except ModuleAttributeError:
            print("Saved model has different architecture.")

    model.networks.print(opt.verbose)

    visualizer = Visualizer(
        display_id=opt.display_id,
        name=opt.name,
        display_port=opt.display_port,
        checkpoints_dir=opt.checkpoints_dir,
    )  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    visualizer.show_hyperparameters(opt.to_dict())
    total_epochs = opt.n_epochs + opt.n_epochs_decay

    if opt.is_wgan:
        critic_batches = []
        critic_gen = ml_data.unpaired_sample_generator(dataset_a, dataset_b)
        for _ in range(opt.wgan_initial_n_critic):
            (a_critic, _), (b_critic, _) = next(critic_gen)
            a_critic = a_critic.to(device)
            b_critic = b_critic.to(device)
            critic_batches.append((a_critic, b_critic))
        model.pre_training(critic_batches)
    else:
        model.pre_training()

    print(f"Start training using {'Wasserstein-CycleGAN' if opt.is_wgan else 'CycleGAN'}")
    try:
        for epoch in range(total_epochs):  # outer loop for all epochs
            epoch_start_time = time.time()  # timer for entire epoch
            epoch_iter = 0  # the number of training iterations in current epoch

            # batch generator
            if opt.is_wgan:
                generator_input_gen = ml_data.unpaired_sample_generator(
                    dataset_a, dataset_b, n_samples=dataset_size // opt.batch_size
                )
                critic_gen = ml_data.unpaired_sample_generator(dataset_a, dataset_b)

                def wgan_generator():
                    with suppress(StopIteration):
                        while True:
                            (a, _), (b, _) = next(generator_input_gen)
                            a = a.to(device)
                            b = b.to(device)
                            critic_batches = []
                            for _ in range(opt.wgan_n_critic):
                                (a_critic, _), (b_critic, _) = next(critic_gen)
                                a_critic = a_critic.to(device)
                                b_critic = b_critic.to(device)
                                critic_batches.append((a_critic, b_critic))
                            yield a, b, critic_batches

                batch_generator = wgan_generator()
            else:
                generator_input_gen = ml_data.unpaired_sample_generator(
                    dataset_a, dataset_b, n_samples=dataset_size // opt.batch_size
                )

                def gan_generator():
                    with suppress(StopIteration):
                        while True:
                            (a, _), (b, _) = next(generator_input_gen)
                            a = a.to(device)
                            b = b.to(device)
                            yield a, b

                batch_generator = gan_generator()

            # Get random permutations of items from both datasets
            for batch in iter(batch_generator):

                iter_start_time = time.time()  # timer for computation per iteration

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                stats = model.do_iteration(*batch)

                if total_iters % (opt.print_freq * opt.batch_size) == 0:
                    losses = stats.get_losses()
                    visuals = stats.get_visuals()
                    time_per_iteration = (time.time() - iter_start_time) / opt.batch_size
                    estimated_time = (
                        total_epochs * dataset_size - total_iters
                    ) * time_per_iteration
                    visualizer.print_current_losses(
                        epoch + 1, epoch_iter, losses, time_per_iteration, estimated_time
                    )
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses
                    )
                    visualizer.display_current_results(visuals)

                if total_iters % (opt.save_freq * opt.batch_size) == 0:
                    model.networks.save(
                        os.path.join(
                            os.path.join(opt.checkpoints_dir, opt.name), "latest_net_"
                        )
                    )

            # update learning rates in the beginning of every epoch.
            model.update_learning_rate()

            path = os.path.join(opt.checkpoints_dir, opt.name)
            model.networks.save(os.path.join(path, "latest_net_"))
            model.networks.save(os.path.join(path, f"{epoch}_net_"))
            visualizer.save_losses_as_image(os.path.join(path, "loss.png"))
            print(f"Saved the model at the end of epoch {epoch}")

            print(
                f"End of epoch {epoch + 1} / {total_epochs} \t"
                f"Time Taken: {time.time()-epoch_start_time} sec"
            )
    except KeyboardInterrupt:
        model.networks.save(
            os.path.join(os.path.join(opt.checkpoints_dir, opt.name), "interrupted_")
        )
