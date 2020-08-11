import os

import yaml

import simulation.utils.machine_learning.data as ml_data
from simulation.utils.machine_learning.cycle_gan.models.cycle_gan_model import CycleGANModel
from simulation.utils.machine_learning.cycle_gan.util import html
from simulation.utils.machine_learning.cycle_gan.util.visualizer import save_images

CONFIG_FILE_PATH = "config.yml"

if __name__ == "__main__":
    config_file = open(CONFIG_FILE_PATH)
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
        transform_properties=tf_properties,
    )  # create datasets for each domain (A and B)

    model = CycleGANModel.from_options(opt)  # create a model given model and other options
    model.setup(
        verbose=opt["verbose"],
        continue_train=opt["continue_train"],
        load_iter=opt["load_iter"],
        epoch=opt["epoch"],
        lr_policy=opt["lr_policy"],
        lr_decay_iters=opt["lr_decay_iters"],
        n_epochs=opt["n_epochs"],
    )
    model.eval()
    # create a website
    web_dir = os.path.join(
        opt["results_dir"], opt["name"], str(opt["epoch"])
    )  # define the website directory
    if opt["load_iter"] > 0:  # load_iter is 0 by default
        web_dir = "{:s}_iter{:d}".format(web_dir, opt["load_iter"])
    print("creating web directory", web_dir)
    webpage = html.HTML(
        web_dir, "Experiment = %s, Epoch = %s" % (opt["name"], opt["epoch"]),
    )
    for i, ((A, A_paths), (B, B_paths)) in enumerate(zip(dataset_a, dataset_b)):
        model.set_input(
            {"A": A, "A_paths": A_paths, "B": B, "B_paths": B_paths}
        )  # unpack data from dataset and apply preprocessing
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        if i % 5 == 0:
            print("processing (%04d)-th image." % i)
        save_images(
            webpage,
            visuals,
            model.image_paths,
            aspect_ratio=opt["aspect_ratio"],
            width=opt["display_winsize"],
        )
    webpage.save()  # save the HTML
