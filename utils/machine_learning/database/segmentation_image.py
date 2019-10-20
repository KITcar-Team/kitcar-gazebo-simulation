"""
This file contains a definition of Segmentation Image
SegmentationImage is a class, which contains information about an input and possibly a mask for training / validating purposes.

Author: Konstantin Ditschuneit
Date: 19.08.2019
"""
import os  # For file paths
import cv2  # Read images
import img.transformer as transformer  # image transformation functions

# Used to get constants (i.e. folder paths) for image locations
from database import connector as con


class SegmentationImage:
    """
    SegmentationImage is a class, which contains information about an input and possibly a mask for training / validating purposes.

    Attributes
    ----------
    key : str
        the hashed image used as a key in the database, used to store the image and its mask (label)
    previous_key : str
        key of previous image
    next_key : str
        key of next image
    small_binary : binary
        binary representation of a small binarized version of the image (to find similar images)
    time_stamp : DateTime
        when image was inserted into database
    type : str
        image type (currently always png)
    dataset_name: str
        name of the dataset, that this image belongs to
    input_path: str
        path of input image
    mask_path: str, optional
        optional path to mask
    """

    def __init__(self, arg, input_folder, dataset_name=None, mask_folder=None, previous_key=None, next_key=None, small_binary=None, time_stamp=None):
        """Initializer

        If the argument `arg` is a tuple, the class is initialized from the arguments of the tuple, otherwise all arguments are taken into account
        sound is used.

        Parameters
        ----------
        arg: string that is the hashed image
             or tuple containing all image variables (e.g. when loaded from database)
        input_folder: str
            path of folder in which the input image is
        mask_path: str, optional
            optional path to folder of mask
        """
        if type(arg) is tuple:
            self.key = arg[0]
            self.previous_key = arg[1]
            self.next_key = arg[2]
            self.small_binary = arg[3]
            self.time_stamp = arg[4]
            self.dataset_name = arg[5]
        else:
            self.key = arg
            self.previous_key = previous_key
            self.next_key = next_key
            self.small_binary = small_binary
            self.time_stamp = time_stamp
            self.dataset_name = dataset_name

        self.type = '.png'

        self.input_path = os.path.join(input_folder,self.key+self.type)

        if mask_folder is None:
            self.mask_path = None
        else:
            self.mask_path = os.path.join(mask_folder,self.key+self.type)

    def load_input_img(self, size=None):
        """
        Returns loaded image

        Parameters
        ----------
        size: [Height,Width],optional
            If specified, the loaded image is resized
        """
        if size is None:
                
            return cv2.imread(self.input_path,0)
        else:
            return transformer.resize(self.load_input_img(), size)


    def load_input_tensor(self, size=None):
        """
        Returns loaded image as pytorch tensor

        Parameters
        ----------
        size: [Height,Width],optional
            If specified, the loaded tensor is resized
        """
        return transformer.image_to_tensor(self.load_input_img(size))

    def load_mask_img(self, size=None):
        """
        Returns loaded mask (or None)

        Parameters
        ----------
        size: [Height,Width],optional
            If specified, the loaded mask is resized
        """
        if self.mask_path is None:
            return None

        if size is None:
            return cv2.imread(self.mask_path)
        else:
            return transformer.resize(self.load_mask_img(), size)


    def load_mask_tensor(self, size=None):
        """
        Returns loaded mask as pytorch tensor (or None)

        Parameters
        ----------
        size: [Height,Width],optional
            If specified, the loaded tensor is resized
        """
        if self.mask_path is None:
            return None

        return transformer.mask_to_class_tensor(self.load_mask_img(size))
