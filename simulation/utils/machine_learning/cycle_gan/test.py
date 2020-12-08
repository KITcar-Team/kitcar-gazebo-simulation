"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.

Example (You need to train models first):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/kitcar --name maps_cyclegan --model cycle_gan

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os

from simulation.utils.machine_learning.cycle_gan.data import create_dataset
from simulation.utils.machine_learning.cycle_gan.models import create_model
from simulation.utils.machine_learning.cycle_gan.options.test_options import TestOptions
from simulation.utils.machine_learning.cycle_gan.util import html
from simulation.utils.machine_learning.cycle_gan.util.visualizer import save_images

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are
    # needed.
    opt.no_flip = (
        True  # no flip; comment this line if results on flipped images are needed.
    )
    opt.display_id = (
        -1
    )  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.eval()
    # create a website
    web_dir = os.path.join(
        opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.epoch)
    )  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = "{:s}_iter{:d}".format(web_dir, opt.load_iter)
    print("creating web directory", web_dir)
    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s" % (opt.name, opt.phase, opt.epoch),
    )
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:
            print("processing (%04d)-th image." % i)
        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
        )
    webpage.save()  # save the HTML
