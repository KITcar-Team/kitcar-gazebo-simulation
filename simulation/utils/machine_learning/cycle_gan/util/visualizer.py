import os
import subprocess
import sys
import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from visdom import Visdom

from simulation.utils.machine_learning.cycle_gan.util.util import mkdir
from . import util

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(
    visuals: dict, destination: str, aspect_ratio: float = 1.0, iteration_count: int = 1,
) -> None:
    """Save images to the disk.

    This function will save images stored in 'visuals'.

    Args:
        destination: the folder to save the images to
        visuals (dict): an ordered dictionary that stores (name, images (either
            tensor or numpy) ) pairs
        aspect_ratio (float): the aspect ratio of saved images
    """
    destination = os.path.join(destination, "images")
    mkdir(destination)
    for label, im_data in visuals.items():
        mkdir(os.path.join(destination, label))
        im = util.tensor2im(im_data)
        save_path = os.path.join(destination, label, f"{iteration_count}.png")
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)


class Visualizer:
    """This class includes several functions that can display/save images and
    print/save logging information.

    It uses a Python library 'visdom' for display.
    """

    def __init__(
        self,
        display_id: int = 1,
        name: str = "kitcar",
        display_port: int = 8097,
        checkpoints_dir: str = "./checkpoints",
    ):
        """Initialize the Visualizer class

        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create a logging file to store training losses

        Args:
            display_id (int): window id of the web display
            name (str): name of the experiment. It decides where to store
                samples and models
            display_port (int): visdom port of the web display
            checkpoints_dir (str): models are saved here
        """
        self.display_id = display_id
        self.name = name
        self.port = display_port
        self.checkpoints_dir = checkpoints_dir
        self.saved = False
        if (
            self.display_id > 0
        ):  # connect to a visdom server given <display_port> and <display_server>
            self.create_visdom_connections(self.port)
            self.vis = Visdom(port=self.port)

        self.log_name = os.path.join(checkpoints_dir, name, "loss_log.txt")
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write("================ Training Loss (%s) ================\n" % now)

    def reset(self) -> None:
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self, port: int) -> None:
        """If the program could not connect to Visdom server, this function will
        start a new server at port < self.port >
        """
        subprocess.Popen(
            ["visdom", "-p", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"Launched Visdom server: http://localhost:{port}")

    def show_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Create a html table with all parameters from the dict and displays it on visdom.

        Args:
            hyperparameters: a dict containing all hyperparameters
        """
        html = (
            "<style>"
            "table"
            "{ border: 1px solid white; width: 500px; height: 200px; text-align: center; border-collapse: collapse; }"
            "table td, table th"
            "{ border: 1px solid #FFFFFF; padding: 3px 2px; }"
            "table tbody td"
            "{font-size: 13px;}"
            "table tr:nth-child(even)"
            "{ background: #D0E4F5; }"
            "table thead"
            "{ background: #0B6FA4; border-bottom: 5px solid #FFFFFF; }"
            "table thead th {"
            "font-size: 17px; font-weight: bold; color: white; text-align: center; border-left: 2px solid white;}"
            "table thead th:first-child"
            "{ border-left: none; }"
            "</style> "
        )
        html += "<h1>Hyperparameters</h1>"
        html += "<table>"
        html += "<thead><tr><th>Key</th><th>Value</th></tr></thead>"
        html += "<tbody>"

        for key, value in hyperparameters.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html += "</tbody></table>"

        self.vis.text(html)

    def display_current_results(self, visuals: dict) -> None:
        """Display current results on visdom.

        Args:
            visuals (dict): dictionary of images to display or save
        """
        if self.display_id > 0:  # show images in the browser using visdom
            # create a table of images.
            title = self.name
            images = []
            idx = 0
            image_numpy = None
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                images.append(image_numpy.transpose([2, 0, 1]))
                idx += 1
            white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
            while idx % 4 != 0:
                images.append(white_image)
                idx += 1
            self.vis.images(
                images,
                nrow=4,
                win=str(self.display_id + 1),
                padding=2,
                opts=dict(title=title + " images"),
            )

    def plot_current_losses(self, epoch: int, counter_ratio: float, losses: dict) -> None:
        """display the current losses on visdom display: dictionary of error
        labels and values

        Args:
            epoch (int): current epoch
            counter_ratio (float): progress (percentage) in the current epoch,
                between 0 to 1
            losses (dict): training losses stored in the format of (name, float)
                pairs
        """
        if not hasattr(self, "plot_data"):
            self.plot_data = {"X": [], "Y": [], "legend": list(losses.keys())}
        self.plot_data["X"].append(epoch + counter_ratio)
        self.plot_data["Y"].append([losses[k] for k in self.plot_data["legend"]])
        self.vis.line(
            X=np.stack([np.array(self.plot_data["X"])] * len(self.plot_data["legend"]), 1),
            Y=np.array(self.plot_data["Y"]),
            opts={
                "title": self.name + " loss over time",
                "legend": self.plot_data["legend"],
                "xlabel": "epoch",
                "ylabel": "loss",
            },
            win=str(self.display_id),
        )

    def save_losses_as_image(self) -> None:
        """This method saves the tracked losses as png file under the checkpoint."""
        # Create figure
        fig = plt.figure()
        # Create sub plot
        ax = fig.add_subplot(1, 1, 1)
        # transpose y data, as self.plot_data['Y'] has the wrong format for matplotlib
        y_data_transposed = [[] for i in range(8)]
        for entry in self.plot_data["Y"]:
            for i, item in enumerate(entry):
                y_data_transposed[i].append(item)

        # Adding all losses to the figure
        for y_data, label in zip(y_data_transposed, self.plot_data["legend"]):
            ax.plot(self.plot_data["X"], y_data, label=label)
            ax.legend()

        # Save plot
        plt.savefig(os.path.join(self.checkpoints_dir, self.name, "loss.png"))

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(
        self, epoch: int, iters: int, losses: dict, t_comp: float
    ) -> None:
        """print current losses on console; also save the losses to the disk

        Args:
            epoch (int): current epoch
            iters (int): current training iteration during this epoch (reset to
                0 at the end of every epoch)
            losses (dict): training losses stored in the format of (name, float)
                pairs
            t_comp (float): computational time per data point (normalized by
                batch_size)
        """
        message = "(epoch: %d, iters: %d, time: %.3f) " % (epoch, iters, t_comp)
        for k, v in losses.items():
            message += "%s: %.3f " % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)  # save the message
