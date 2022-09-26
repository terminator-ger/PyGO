import numpy as np
import cv2
from pygo.utils.debug import DebugInfo, DebugInfoProvider
from pygo.utils.typing import Image, B3CImage, Mask
import logging

class MotionDetection(DebugInfoProvider):
    def __init__(self, img: B3CImage) -> None:
        super().__init__()
        self.lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        img = cv2.resize(img, None, fx=0.25,fy=0.25)
        self.p0 = cv2.goodFeaturesToTrack(img, mask = None, **self.feature_params)
        self.imgLast = img
        self.hist = 0

    def hasMotion(self, img: B3CImage) -> bool:
        img = cv2.resize(img, None, fx=0.25,fy=0.25)
        if img.shape != self.imgLast.shape:
            #first iteration after vp detect
            self.imgLast = img
            return True

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.imgLast, img, self.p0, None, **self.lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]
        else:
            return True
        
        motion = np.max(np.abs(good_new-good_old))
        self.p0 = good_new.reshape(-1,1,2)
        self.imgLast = img

        if motion >= 0.2:
            # if we detect wiggle reset the frame counter
            self.hist = self.hist // 2
            return True
        else: 
            if self.hist < 10:
                # block for at least 10 frames
                self.hist += 1
                return True
            else:
                return False



class MotionDetectionMOG2(DebugInfoProvider):
    def __init__(self, img: B3CImage, resize:bool = True) -> None:
        super().__init__()
        self.resize=resize
        if self.resize:
            img = cv2.resize(img, None, fx=0.25,fy=0.25)
        self.imgLast = img
        self.hist = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.motion_active = False

    def hasMotion(self, img: B3CImage) -> bool:
        if self.resize:
            img = cv2.resize(img, None, fx=0.25,fy=0.25)
        fgmask = self.fgbg.apply(img)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)

        if not self.motion_active and fgmask.sum() > 10:
            # hand onto of board
            self.motion_active = True
            self.hist = 0
            return True

        if self.motion_active and fgmask.sum() == 0:
            if self.hist < 5:
                self.hist += 1
                return True
            else:
                # hand out of board
                self.motion_active = False
                self.hist = 0
                logging.debug('No Motion')
                return False

        return True

    def getMask(self, img: B3CImage) -> Mask:
        if self.resize:
            img = cv2.resize(img, None, fx=0.25,fy=0.25)
        fgmask = self.fgbg.apply(img)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        return fgmask