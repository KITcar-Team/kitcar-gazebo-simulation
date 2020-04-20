"""
This file contains a definition of all different colorclasses used in
the gazebo segmentation and ColorClass<->Rgb conversion methods.
Colors with the same grey-value are used to automatically label road objects,
while still appearing the same to dr_drift's greyscale camera!

Author: Konstantin Ditschuneit
Date: 19.08.2019
"""
from enum import Enum  # Needed for ColorClass definition
import cv2  # Used for color conversions
import numpy as np  # Arrays

SEGMENTATION_ENABLED = True  # Specifies if segmentation is used in rendering
NUMBER_OF_CLASSES = 9  # Total number of colorclasses


class ColorClass(Enum):
    """
    Contains all ColorClasses used in segmentation, and a corresponding index
    """

    NONE = -1
    DEFAULT = 0
    SIDE_LINE = 1
    MIDDLE_LINE = 2
    PARKING = 3
    START_LINE = 4
    STOP_LINE = 5
    BLOCKED_AREA = 6
    ZEBRA_CROSSING = 7
    TRAFFIC_SIGN = 8
    PARKING_SPOT_X = 9


def color_vec():
    """
    Returns all rgb Values of all ColorClasses (excluding None) as a list
    """
    return list(map(lambda x: rgb(ColorClass(x)), range(0, NUMBER_OF_CLASSES)))


def rgb(color_class):
    """
    Calculates the rgb-color of given color class
    Args:
        color_class: class of color as ColorClass enum object
    Returns:
        r,g,b color values from 0 to 255
    """
    if not SEGMENTATION_ENABLED:
        return 1, 1, 1  # Return white, if segmentation is not used

    k = color_class.value  # Number of the class

    # if k < 1 or k > 2:
    #    k = 0

    h = 180 * float(k / NUMBER_OF_CLASSES)  # hue
    s = 0.5 * 255  # saturation
    light = 0.8 * 255  # value: brightness

    if k == ColorClass.DEFAULT.value:  # white
        s = 0
    hls = np.uint8([[[h, light, s]]])  # Create hsv array

    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)  # Convert to rgb

    r = rgb[0][0][0] / 255
    g = rgb[0][0][1] / 255
    b = rgb[0][0][2] / 255

    # print("k:", k, "h:", h, "RGB:", r, g, b)

    return r, g, b  # return rgb


def class_from_color(r, g, b):
    """
    Calculates the color class closest to the given rgb
    Args:
        r,g,b color values
    Returns:
        ColorClass
    """

    rgb = np.uint8([[[r, g, b]]])  # rgb array as ints

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)  # convert to hsv

    h = hsv[0][0][0]
    light = hsv[0][0][1]
    s = hsv[0][0][2]

    if light == 0:
        return ColorClass.NONE  # Black
    if s == 0:
        return ColorClass.DEFAULT  # Pure White

    k = h / 180 * NUMBER_OF_CLASSES  # Otherwise calculate from h value

    # print(h,s,v)

    return ColorClass(round(k) % NUMBER_OF_CLASSES)
