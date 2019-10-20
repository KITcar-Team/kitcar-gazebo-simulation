"""
This file contains functions used to resize images and transform images and masks (segmentation labels) to pytorch tensors

Pytorch is used for machine learning purposes and images used for training / validating / testing must be represented as pytorch tensors.

Author: Konstantin Ditschuneit
Date: 19.08.2019
"""
import torch  # Contains the tensors
import numpy as np  # Arrays
import cv2  # Used to resize images
import color_classes as cc  # Used to specify color class from pixel in image

import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

def image_to_tensor(image, mean=0, std=1.):
    """
    Normalizes the image and transforms it to a tensor
    Args:
        image (np.ndarray): A RGB array image
        mean: The mean of the image values
        std: The standard deviation of the image values

    Returns:
        tensor: A Pytorch tensor of shape (C, H, W) where C is the number of color channels (3 for rgb)
    """

    image = rand_trafo(image.astype(np.float32))
    image = (image - mean) / std  # Normalize

    if len(image.shape) == 2: # grayscale image has different shape
        image = np.array([image])  # Move into (C, H, W) shape c=1
    elif len(image.shape)== 3:
        image = image.transpose((2, 0, 1))  # Move into (C, H, W) shape

    tensor = torch.from_numpy(image)  # Convert to pytorch tensor
    return tensor



def mask_to_class_tensor(mask):
    """
    Transforms a mask to a tensor
    Args:
        mask (np.ndarray): A greyscale mask array
    Returns:
        tensor: A Pytorch tensor of shape (N,H,W) where N is the number of classes
    """
    N = cc.NUMBER_OF_CLASSES  # Get total number of segmentation classes
    H = mask.shape[0]
    W = mask.shape[1]

    tensor = np.zeros([N, H, W])

    # Go through all pixels and create tensor with value 1 at position [cc.class_from_color, x, y]
    for y in range(0, mask.shape[0]-1):
        for x in range(0, mask.shape[1]-1):
            # Will fail for ColorClass.None
            tensor[cc.class_from_color(*mask[y][x]).value][y][x] = 1

    return torch.from_numpy(tensor)


def resize(img, new_size):
    """
        Resize an image to the spezified new size
    Args:
        img (np.ndarray): image to resize
        new_size (tuple): The size as tuple (h, w)

    Returns:
        Image: The resized image
    """
    return cv2.resize(img, dsize=(new_size[0], new_size[1]), interpolation=cv2.INTER_AREA)

def rand_trafo(img):
    r = random.randint(1,7)

    if r==2:
        return sk.util.random_noise(img)
    else:
        return img
    