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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def toColorImage(image):
    '''
    Reduces the image to a gray image
    Does NOT change Byte <-> Float, image is left in original datatype
    :param image:
    :return:
    '''
    if len(np.shape(image)) == 2:
         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image



def load_training_data(classes):
    x = []
    y = []

    for folder in os.listdir('data'):
        if int(folder) in classes:
            for file in os.listdir(os.path.join('data', folder)):
                img = io.imread(os.path.join('data', folder, file))
                img = toColorImage(img)
                img = cv2.resize(img, (32,32))
                img = np.array(img)
                x.append(img)
                f = int(folder.split('.png')[0])
                y.append(f)

    return x, y
