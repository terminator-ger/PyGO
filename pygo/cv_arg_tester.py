from __future__ import print_function
from __future__ import division
from typing import Tuple, TypeVar, Callable
import matplotlib.pyplot as plt
import cv2 
import pdb
from pygo.utils.image import toCMYKImage, toByteImage, toGrayImage, toColorImage, toHSVImage
import numpy as np
from functools import partial
import math
from Webcam import Webcam

T = TypeVar('T', int, float)

class Argument:
    name: str
    value: int
    range: Tuple[float,float]
    scaling: Callable
    unscale: Callable
    _dtype: T

    def __init__(self, name, range, value):
        self.name = name
        self.range = range
        self.scaling = lambda x: ((x-range[0]) / (range[1]-range[0]) * 100)
        self.unscale = lambda x: (x/100*(range[1]-range[0])+range[0])
        
        if isinstance(value, int):
            self._dtype = int
        elif isinstance(value, float):
            self._dtype = float

        self.value = int(self.scaling(value))
        print(value)
        print(range)
        print(self.value)
        print(self.unscale(self.value))
    
class CV2ArgTester:
    def __init__(self, img):
        self.args = {}
        self.win_title = 'CV2ArgTester'
        self.win = cv2.namedWindow(self.win_title)
        self.img = img
        self.img_ = None

    def set(self, name: str, value):
        self.args[name].value = value
        #self.args[name].scaling(value)

    def get(self, name: str) -> float:
        return self.args[name]._dtype(self.args[name].unscale(self.args[name].value))

    def addArgument(self, name: str, value, range: Tuple[float,float]):
        self.args[name] = Argument(value, range, value)
        cv2.createTrackbar(name,
                            self.win_title , 
                            self.args[name].value,
                            100,
                            self.update(name))


    def update(self, name, value=None):
        def partial_update(value):
            self.set(name, value)
            self.redraw()
            self.show()
        return partial_update

    def redraw(self):
        raise NotImplementedError()

    def show(self):
        raise NotImplementedError()

class Hough(CV2ArgTester):
    def redraw(self):
       #img = cv2.Canny(toGrayImage(img),50,150,apertureSize = 3)

        _, img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        img_ = img.copy()
        
        lines = cv2.HoughLines(img, 
                                rho=        self.get('rho'), 
                                theta=      self.get('theta'), 
                                threshold=  self.get('threshold'))
                                #srn=        self.get('srn'), 
                                #stn=        self.get('stn')) 
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                # convert both to positive values -> map to first quadrant
                if rho < 0:
                    continue
                a = np.abs(math.cos(theta))
                b = np.abs(math.sin(theta))
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(img_, pt1, pt2, (255,255,255), 1, cv2.LINE_AA)
        self.img_ = img_


    def show(self):
        print('____________________________')
        for k in self.args:
            print('{} : {}'.format(k, self.get(k)))
        if self.img_ is not None:
            cv2.imshow('CV2ArgTester', self.img_)
        else:
            cv2.imshow('CV2ArgTester', self.img)
        cv2.waitKey(0)


if __name__ == '__main__':
    cam = Webcam()
    _, img = cam.read()
    img = toHSVImage(img)[:,:,2]
 
    argtest = Hough(img)
    argtest.addArgument('rho', 1, (1, 10))
    argtest.addArgument('theta', (np.pi/180*1), ((np.pi/180*1),(np.pi/180*30)))
    argtest.addArgument('threshold', 150, (1,500))
    argtest.addArgument('srn', 0, (0,10))
    argtest.addArgument('stn', 0, (0,10))
    argtest.show()

