"""Script for translating a simulated video to a real video

Once you have trained your model with train.py, you can use this script to translate a video. It will load a saved
model from '--checkpoints_dir', the input video from "in.mp4" and outputs the translated video to out.mp4.

Example (You need to train models first):
    Translate a video using a CycleGAN model:
        python convert_video.py

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import shutil

import cv2

from simulation.utils.machine_learning.cycle_gan.util import util
from simulation.utils.machine_learning.cycle_gan.data import create_dataset
from simulation.utils.machine_learning.cycle_gan.models import create_model
from simulation.utils.machine_learning.cycle_gan.options.test_options import TestOptions
from simulation.utils.machine_learning.cycle_gan.util.util import mkdir

FPS = 30
TRANSLATED_IMAGES_FOLDER = "frames"
INPUT_VIDEO_FILE = "in.mp4"
OUTPUT_VIDEO_FILE = "out.mp4"


def extract_images(filename):
    folder = "testB"
    mkdir(folder)
    mkdir("testA")
    video_capture = cv2.VideoCapture(filename)
    count = 0
    while True:
        success, image = video_capture.read()
        if not success:
            break
        cv2.imwrite(
            os.path.join(folder, "frame{:d}.jpg".format(count)), image
        )  # save frame as JPEG file
        if count == 1:
            cv2.imwrite(
                os.path.join("testA", "frame{:d}.jpg".format(count)), image
            )  # save frame as JPEG file
        count += 1
        print("Extracted images %04d" % count)
    print("{} images are extacted in {}.".format(count, folder))


if __name__ == "__main__":
    extract_images(INPUT_VIDEO_FILE)
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.dataroot = "./"
    opt.name = "pre-training"
    opt.direction = "BtoA"
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line
    # if results on randomly chosen images are needed.
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
    mkdir(TRANSLATED_IMAGES_FOLDER)
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:
            print("processing (%04d)-th image." % i)
        util.save_image(
            util.tensor2im(visuals["fake_B"]),
            os.path.join(TRANSLATED_IMAGES_FOLDER, "%d.png" % i),
        )

    images = [img for img in os.listdir(TRANSLATED_IMAGES_FOLDER) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(TRANSLATED_IMAGES_FOLDER, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        OUTPUT_VIDEO_FILE, cv2.VideoWriter_fourcc(*"MP4V"), FPS, (width, height)
    )

    for image in images:
        video.write(cv2.imread(os.path.join(TRANSLATED_IMAGES_FOLDER, image)))

    cv2.destroyAllWindows()
    video.release()
    shutil.rmtree(TRANSLATED_IMAGES_FOLDER)
    shutil.rmtree("testA")
    shutil.rmtree("testB")
