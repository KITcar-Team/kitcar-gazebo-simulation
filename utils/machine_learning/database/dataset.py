"""
This file contains SegmentationImageDataset, which inherits torch.utils.data.dataset and is used to load images for training / testing of networks

Author: Konstantin Ditschuneit
Date: 19.08.2019
"""
import numpy as np  # arrays
import torch.utils.data as data  # Torch dataset
import os  # Filepaths
import cv2  # Open images


# Reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L66
class SegmentationImageDataset(data.Dataset):
    def __init__(self, segmentation_images, img_resize=(320, 160)):
        """
            A dataset loader taking a segmentation images as argument and return
            as them as tensors from getitem()

            Parameters
            ----------
            segmentation_images: list of SegmentationImage
                images contained in the dataset

            img_resize: Size which images will be resized to
        """

        self.segmentation_images = segmentation_images

        self.img_size = img_resize

    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple (torch.tensor, torch.tensor): (image, target) where target has shape (N,H,W) and N= number of classes
        """
        segmentation_image = self.segmentation_images[index]  # get segmentation image

        img_tensor = segmentation_image.load_input_tensor(
            self.img_size)  # load input tensor

        mask_tensor = segmentation_image.load_mask_tensor(
            self.img_size)  # Load mask tensor

        if mask_tensor is None:
            return img_tensor.float(), None
        else:
            return img_tensor.float(), mask_tensor.float()  # return as tuple

    def __len__(self):
        """
            Returns:
                int: number of images in dataset
        """
        return len(self.segmentation_images)
