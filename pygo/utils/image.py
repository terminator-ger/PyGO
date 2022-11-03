import os
from re import A
import cv2
import numpy as np
from enum import Enum
from skimage import io

#
#
#

class Color(Enum):
    WHITE = 0
    BLACK = 1
    NONE  = 2


def N2C(c):
    if c == 0:
        c_str = 'W' 
    elif c == 1:
        c_str = 'B' 
    elif c == 2:
        c_str = 'E' 
    return c_str

def C2N(c):
    if c == 'W':
        return 0
    elif c == 'B':
        return 1
    elif c == 'E':
        return 2

def toNP(x):
    return x.detach().cpu().numpy()

def toBoolImage(image):
    img = toByteImage(image)
    return img.astype(np.bool)

def toByteImage(image):
    '''
    Returns the uint8 representation of an image
    :param image:
    :return:
    '''
    if np.max(image) <= 1.0 or np.max(image) == 0:
        image = (image*255)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image

def toHSVImage(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return img

def toFloatImage(image):
    '''
    Returns the Float32 representation of an image
    :param image:
    :return:
    '''
    if np.max(image) > 1 or np.max(image) == 0:
        image = (image/255).astype(np.float32)
    return image

def toDoubleImage(image):
    '''
    Returns the Float32 representation of an image
    :param image:
    :return:
    '''
    if np.max(image) > 1 or np.max(image) == 0:
        image = (image/255).astype(np.double)
    return image


def toGrayImage(image):
    '''
    Reduces the image to a gray image
    Does NOT change Byte <-> Float, image is left in original datatype
    :param image:
    :return:
    '''
    if len(np.shape(image))>2:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return image.copy()


def toColorImage(image):
    '''
    Reduces the image to a gray image
    Does NOT change Byte <-> Float, image is left in original datatype
    :param image:
    :return:
    '''
    if len(np.shape(image)) == 2:
         ret = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        ret = image.copy()
    return ret


# Load image
def toCMYKImage(image):
    bgrdash = toByteImage(image).astype(float)/255.

    # Calculate K as (1 - whatever is biggest out of Rdash, Gdash, Bdash)
    K = 1 - np.max(bgrdash, axis=2)
    # Calculate C
    C = (1-bgrdash[...,2] - K)/((1-K)+1e-12)

    # Calculate M
    M = (1-bgrdash[...,1] - K)/((1-K)+1e-12)

    # Calculate Y
    Y = (1-bgrdash[...,0] - K)/((1-K)+1e-12)

    # Combine 4 channels into single image and re-scale back up to uint8
    CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)
    return CMYK

def toYUVImage(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def cmykToBGR(image, cmyk_scale=255, rgb_scale=255):
    c,m,y,k = cv2.split(image)
    r = np.uint8(rgb_scale * (1.0 - c / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale)))
    g = np.uint8(rgb_scale * (1.0 - m / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale)))
    b = np.uint8(rgb_scale * (1.0 - y / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale)))
    return cv2.merge([b,g,r])
