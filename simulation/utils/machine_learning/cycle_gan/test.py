import argparse
import os

import yaml

import simulation.utils.machine_learning.data as ml_data
from simulation.utils.machine_learning.cycle_gan.models.cycle_gan_model import CycleGANModel
from simulation.utils.machine_learning.cycle_gan.util.visualizer import save_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read config file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="path to config file where all parameters are stored",
    )
    config_file_path = parser.parse_args().config

    with open(config_file_path) as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)

    opt = {**configs["base"], **configs["test"]}

    tf_properties = {
        "load_size": opt["load_size"],
        "crop_size": opt["crop_size"],
        "preprocess": opt["preprocess"],
        "mask": opt["mask"],
    }
    dataset_a, dataset_b = ml_data.load_unpaired_unlabeled_datasets(
        opt["dataset_a"],
        opt["dataset_b"],
        batch_size=1,
        serial_batches=True,
        num_threads=0,
        grayscale_A=(opt["input_nc"] == 1),
        grayscale_B=(opt["output_nc"] == 1),
        max_dataset_size=opt["max_dataset_size"],
        transform_properties=tf_properties,
    )  # create datasets for each domain (A and B)

    model = CycleGANModel.from_options(
        **opt
    )  # create a model given model and other options
    model.setup(
        verbose=opt["verbose"],
        continue_train=False,
        load_iter=opt["load_iter"],
        epoch=opt["epoch"],
    )
    model.eval()
    for i, ((A, A_paths), (B, B_paths)) in enumerate(zip(dataset_a, dataset_b)):
        model.set_input(
            {"A": A, "A_paths": A_paths, "B": B, "B_paths": B_paths}
        )  # unpack data from dataset and apply preprocessing
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        if i % 5 == 0:
            print("processing (%04d)-th image." % i)
        save_images(
            visuals=visuals,
            destination=os.path.join(opt["results_dir"], opt["name"]),
            aspect_ratio=opt["aspect_ratio"],
            iteration_count=i,
        )
