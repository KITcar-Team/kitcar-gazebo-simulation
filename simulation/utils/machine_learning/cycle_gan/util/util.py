"""This module contains simple helper functions """
from __future__ import print_function

import os

import numpy as np
import torch
from PIL import Image


def tensor2im(input_image, img_type=np.uint8, to_rgb: bool = True):
    """"Converts a Tensor array into a numpy image array.

    Args:
        input_image (np.ndarray): the input image tensor array
        img_type (np.integer): the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1 and to_rgb:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (
            (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        )  # post-processing: transpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(img_type)


def save_image(image_numpy: np.ndarray, image_path: str, aspect_ratio: float = 1.0) -> None:
    """Save a numpy image to the disk

    Args:
        image_numpy (np.ndarray): input numpy array
        image_path (str): the path of the image
        aspect_ratio (float): the aspect ratio of the resulting image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def mkdir(path: str) -> None:
    """create a single empty directory if it didn't exist

    Args:
        path (str): a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
