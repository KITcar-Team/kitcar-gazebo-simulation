"""A modified image folder class.

We modify the official PyTorch image folder
(https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import os
import os.path
from typing import List

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    """Check if a file is an image.

    Args:
        filename: the file name to check
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_images(dir: str, max_dataset_size=float("inf")) -> List[str]:
    """Recursively search for images in the given directory.

    Args:
        dir: the directory of the dataset
        max_dataset_size: the maximum amount of images to load
    """
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in os.walk(dir):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return sorted(images[: min(max_dataset_size, len(images))])
