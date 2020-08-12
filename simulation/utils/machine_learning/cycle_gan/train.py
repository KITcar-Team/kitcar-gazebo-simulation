import argparse
import time

import yaml

import simulation.utils.machine_learning.data as ml_data
from simulation.utils.machine_learning.cycle_gan.models.cycle_gan_model import CycleGANModel
from simulation.utils.machine_learning.cycle_gan.util.visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read config file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="path to config file where all parameters are stored",
    )
    config_file_path = parser.parse_args().config

    config_file = open(config_file_path)
    configs = yaml.load(config_file, Loader=yaml.FullLoader)

    opt = {**configs["base"], **configs["train"]}

    tf_properties = {
        "load_size": opt["load_size"],
        "crop_size": opt["crop_size"],
        "preprocess": opt["preprocess"],
        "mask": opt["mask"],
    }
    dataset_a, dataset_b = ml_data.load_unpaired_unlabeled_datasets(
        opt["dataset_a"],
        opt["dataset_b"],
        batch_size=opt["batch_size"],
        serial_batches=opt["serial_batches"],
        num_threads=opt["num_threads"],
        grayscale_A=(opt["input_nc"] == 1),
        grayscale_B=(opt["output_nc"] == 1),
        max_dataset_size=opt["max_dataset_size"],
        transform_properties=tf_properties,
    )  # create datasets for each domain (A and B)

    dataset_size = min(
        len(dataset_a), len(dataset_b)
    )  # get the number of images in the dataset.
    print("The number of training images = %d" % dataset_size)

    model = CycleGANModel.from_options(opt)  # create a model given model and other options
    model.setup(
        verbose=opt["verbose"],
        continue_train=opt["continue_train"],
        load_iter=opt["load_iter"],
        epoch=opt["epoch"],
        lr_policy=opt["lr_policy"],
        lr_decay_iters=opt["lr_decay_iters"],
        n_epochs=opt["n_epochs"],
    )  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(
        display_id=opt["display_id"],
        is_train=True,
        no_html=opt["no_html"],
        display_winsize=opt["display_winsize"],
        name=opt["name"],
        display_port=opt["display_port"],
        display_server=opt["display_server"],
        display_env=opt["display_env"],
        checkpoints_dir=opt["checkpoints_dir"],
    )  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    for epoch in range(
        opt["epoch_count"], opt["n_epochs"] + opt["n_epochs_decay"] + 1
    ):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = (
            0  # the number of training iterations in current epoch, reset to 0 every epoch
        )
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        # Get random permutations of items from both datasets
        for (A, A_paths), (B, B_paths) in zip(
            dataset_a, dataset_b
        ):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += opt["batch_size"]
            epoch_iter += opt["batch_size"]
            model.set_input(
                {"A": A, "A_paths": A_paths, "B": B, "B_paths": B_paths}
            )  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if (
                total_iters % opt["display_freq"] == 0
            ):  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt["update_html_freq"] == 0
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result
                )

            if (
                total_iters % opt["print_freq"] == 0
            ):  # print training losses and save logging information to the disk
                t_data = iter_start_time - iter_data_time
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt["batch_size"]
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt["display_id"] > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses
                    )

            if (
                total_iters % opt["save_latest_freq"] == 0
            ):  # cache our latest model every <save_latest_freq> iterations
                print(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                save_suffix = "iter_%d" % total_iters if opt["save_by_iter"] else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        if (
            epoch % opt["save_epoch_freq"] == 0
        ):  # cache our model every <save_epoch_freq> epochs
            print(
                "saving the model at the end of epoch %d, iters %d" % (epoch, total_iters)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (
                epoch,
                opt["n_epochs"] + opt["n_epochs_decay"],
                time.time() - epoch_start_time,
            )
        )
