"""
This file contains functions, to hash and compare images

Author: Konstantin Ditschuneit
Date: 19.08.2019
"""
import hashlib # Hashing
import img.transformer as transformer #resize
import cv2 #Image manipulation
import base64 #Encode buffer to string

def hash_image(img):
    """
        Returns md5 hash of image

        Parameters
        ----------
        img (array)
            Loaded image as array
        """
    buffer = cv2.imencode('.png', img)[1] #Encode and select buffer
    png_as_text = base64.b64encode(buffer) # encode to text
    
    return hashlib.md5(png_as_text).hexdigest() #Return md5 hash


def small_bin(img, size=[20,40]):
    """
        Returns binary string of resized (small) and binarized image

        Parameters
        ----------
        img: image

        size: [Height,Width],default = [20,40]
            the loaded image is resized
        """
    cropped = cv2.threshold(transformer.resize(img,size),10,255,cv2.THRESH_BINARY)[1] # Resize and binarize image
    
    #Transform into string
    s = ''
    for x in (cropped/255).astype(int).flatten():
        s += str(x)
    
    return s #Return string, consistent of 0s and 1s


