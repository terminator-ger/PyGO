import cv2
import numpy as np

from pygo.utils.debug import DebugInfo, DebugInfoProvider, Timing
from pygo.utils.image import toByteImage, toColorImage, toGrayImage
from pygo.utils.typing import Image, B3CImage, Mask
from pygo.Signals import *
from pygo.classifier import CircleClassifier

from enum import Enum, auto

import logging


class debugkeys(Enum):
    Motion = auto()
    BoardMotion = auto()



class MotionDetectionMOG2(DebugInfoProvider):
    '''
        Detects Motion between two frames and blocks the classification algorithm
        based on a simple threshold with an histersis to prevent early unocking
    
    '''
    def __init__(self, img: B3CImage, classifier: CircleClassifier, resize:bool = True) -> None:
        DebugInfoProvider.__init__(self)
        
        self.resize=resize
        if self.resize:
            self.f = 0.25
            img = cv2.resize(img, None, fx=self.f,fy=self.f)
        else:
            self.f = 1
        self.imgLast = img
        self.hist = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2(400, detectShadows=False)
        self.fgbg_border = cv2.createBackgroundSubtractorMOG2(400, detectShadows=False)
        self.motion_active = False

        # reuse the classifiers hidden intersection module
        self.hiddenIntersections = classifier
        self.last_hidden_count = 0

        self.settings = {
            'MotionDetectionBoard' : 0.6,
            'MotionDetectionBorder' : 0.1,
        }

        for key in debugkeys:
            self.available_debug_info[key.name] = False
        Signals.subscribe(OnSettingsChanged, self.settings_updated)
        Signals.subscribe(OnGridSizeUpdated, self.grid_size_updated)
        Signals.subscribe(GameNew, self.reset)
        Signals.subscribe(GameHandicapMove, self.update_hidden_count_with_handicap)

        self.stone_area = None
        self.tresh = self.stone_area if self.stone_area is not None else 1
        self.bs = 0

    def reset(self, *args):
        self.last_hidden_count = None

    def update_hidden_count_with_handicap(self, args):
        n = args[0]
        self.last_hidden_count = n


    def grid_size_updated(self, args):
        width = args[0]
        height = args[1]
        radius = min(width, height) / 2
        area = (radius * self.f)**2 * np.pi
        self.stone_area = area
        self.tresh = area
        self.bs = int(min(width, height) * self.f)

    def settings_updated(self, args):
        new_settings = args[0]
        for k in self.settings.keys():
            self.settings[k] = new_settings[k].get()

    def getHiddenIntersectionCount(self, img:B3CImage) -> bool:
        if self.resize:
            F = 1.0
            img_count = cv2.resize(img, None, fx=F,fy=F)
            img = cv2.resize(img, None, fx=self.f,fy=self.f)
        else:
            F = 1.0

        if self.hiddenIntersections is not None:
            hidden_count = self.hiddenIntersections.get_hidden_intersection_count(img_count, scale=F)
            return hidden_count
        else:
            raise RuntimeError('Missing ref to alg make sure to pass \
                                a circle classifier instance to the motiondetection')

    def hasNoMotion(self, img: B3CImage) -> bool:
        return self._hasNoMotion(img)
    
    def _hasNoMotionSimple(self, img: B3CImage) -> bool:
        if self.last_hidden_count is None:
            self.last_hidden_count = self.getHiddenIntersectionCount(img)
        hidden_count = self.getHiddenIntersectionCount(img)
        if (hidden_count - self.last_hidden_count > 2):
            self.motion_active = True
            return False

        if (self.motion_active and \
            (hidden_count - self.last_hidden_count <= 2)):
            self.last_hidden_count = hidden_count
            self.motion_active = False
            logging.debug('No Motion')
            return True

        return False


    def _hasNoMotion(self, img: B3CImage) -> bool:
        if self.resize:
            F = 1.0
            img_count = cv2.resize(img, None, fx=F,fy=F)
            img = cv2.resize(img, None, fx=self.f,fy=self.f)
        else:
            F = 1.0

        bmask = self.fgbg_border.apply(img, self.settings['MotionDetectionBorder'])
        fgmask = self.fgbg.apply(img, self.settings['MotionDetectionBoard'])

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        #bmask = fgmask.copy()
        bmask = cv2.morphologyEx(bmask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        bmask[self.bs:-self.bs, self.bs:-self.bs] = 0
        fgmask[:self.bs] = 0
        fgmask[:,:self.bs] = 0
        fgmask[-self.bs:] = 0
        fgmask[:,-self.bs:] = 0

        idx = np.argwhere(fgmask > 0)
        if len(idx) > 0:
            x_min = np.min(idx[:,0])
            x_max = np.max(idx[:,0])
            y_min = np.min(idx[:,1])
            y_max = np.max(idx[:,1])
            dx = x_max-x_min
            dy = y_max-y_min
            area = dx*dy
        else:
            area = 0

        bmask_disp = cv2.resize(bmask, None, fx=4,fy=4)
        mask_disp = cv2.resize(fgmask, None, fx=4,fy=4)
        mm = np.dstack((bmask_disp, mask_disp, bmask_disp))
        self.showDebug(debugkeys.Motion, mm)

        val = fgmask > 0
        val = val.sum()
        bval = bmask > 0
        bval = bval.sum()

        if (not self.motion_active and 
                val > .8*self.tresh and 
                bval > 0):# and
                #hidden_count -self.last_hidden_count > 2):
            # hand onto of board
            self.motion_active = True
            self.hist = 0
            return False

        if (self.motion_active and bval <=  3 and 
                area < 3*self.stone_area):# and
                #(hidden_count - self.last_hidden_count <= 2)):
            #self.last_hidden_count = hidden_count
            self.motion_active = False
            self.hist = 0
            logging.debug('No Motion')
            return True

        return False

    def getMask(self, img: B3CImage) -> Mask:
        if self.resize:
            img = cv2.resize(img, None, fx=self.f,fy=self.f)
        fgmask = self.fgbg.apply(img)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        return fgmask
