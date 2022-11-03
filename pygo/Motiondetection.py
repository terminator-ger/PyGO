from distutils.log import debug
import numpy as np
import cv2


from pygo.utils.debug import DebugInfo, DebugInfoProvider, Timing
from pygo.utils.image import toByteImage, toGrayImage
from pygo.utils.typing import Image, B3CImage, Mask
from pygo.Signals import OnBoardDetected, OnSettingsChanged, Signals, OnBoardMoved
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import bgsubcnt
import pdb
from enum import Enum, auto

import logging
import mediapipe as mp

class debugkeys(Enum):
    Motion = auto()
    BoardMotion = auto()


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

    def hasNoMotion(self, img: B3CImage) -> bool:
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


class MotionDetectionBorderMask(DebugInfoProvider):
    '''
        We only use the Borders around the go board and compare with a reference image
        to detect the presence of an arm
        Attention: We need a hand free image during initialization as well as
                   enough space around the board in the image
    '''
    def __init__(self)-> None:
        super().__init__()
        self.ref = None
        self.hist = 0
        self.motion_active = False
        OnBoardDetected.subscribe(self.initRefImage)
    
    def initRefImage(self, img, corners, H) -> None:
        img = toGrayImage(img)
        self.ref = np.vstack((img[:10].T, img[-10:].T, img[:,:10], img[:,-10:]))
        # black out center of the image

    def hasMotion(self, img: B3CImage) -> bool:
        if self.ref is None:
            return True

        img = toGrayImage(img)
        img = np.vstack((img[:10].T, img[-10:].T, img[:,:10], img[:,-10:]))
        (score, diff) = structural_similarity(self.ref, img, full=True)
        print("Image Similarity: {:.4f}%".format(score * 100))

        if not self.motion_active and score < 0.9:
            # hand onto of board
            self.motion_active = True
            self.hist = 0
            return True

        if self.motion_active and score > 0.9:
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



class MotionDetectionMOG2(DebugInfoProvider):
    '''
        Detects Motion between two frames and blocks the classification algorithm
        based on a simple threshold with an histersis to prevent early unocking
    
    '''
    def __init__(self, img: B3CImage, resize:bool = True) -> None:
        DebugInfoProvider.__init__(self)
        
        self.resize=resize
        if self.resize:
            img = cv2.resize(img, None, fx=0.25,fy=0.25)
        self.imgLast = img
        self.hist = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2(240, detectShadows=True)
        self.motion_active = False
        self.settings = {
            'MotionDetectionFactor' : -1,
        }
        for key in debugkeys:
            self.available_debug_info[key.name] = False
        Signals.subscribe(OnSettingsChanged, self.settings_updated)

    def settings_updated(self, args):
        new_settings = args[0]
        for k in self.settings.keys():
            self.settings[k] = new_settings[k].get()

    def hasNoMotion(self, img: B3CImage) -> bool:
        if self.resize:
            img = cv2.resize(img, None, fx=0.25,fy=0.25)
        fgmask = self.fgbg.apply(img, self.settings['MotionDetectionFactor'])
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)

        self.showDebug(debugkeys.Motion, fgmask)

        if not self.motion_active and fgmask.sum() > 10:
            # hand onto of board
            self.motion_active = True
            self.hist = 0
            return False

        if self.motion_active and fgmask.sum() <= 0:
            #if self.hist < 1:
            #    self.hist += 1
            #    return False
            #else:
                # hand out of board
                self.motion_active = False
                self.hist = 0
                logging.debug('No Motion')
                return True

        return False

    def getMask(self, img: B3CImage) -> Mask:
        if self.resize:
            img = cv2.resize(img, None, fx=0.25,fy=0.25)
        fgmask = self.fgbg.apply(img)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        return fgmask

class BoardShakeDetectionMOG2(DebugInfoProvider):
    '''
        Detects Motion between two frames and emits a reinitialization signal
        based on a simple threshold 
    
    '''
    def __init__(self, resize:bool = True) -> None:
        DebugInfoProvider.__init__(self)
        
        self.resize=resize
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.settings = {
            'MotionDetectionFactor' : 0.1,
        }
        for key in debugkeys:
            self.available_debug_info[key.name] = False
        #Signals.subscribe(OnSettingsChanged, self.settings_updated)

    #def settings_updated(self, args):
    #    new_settings = args[0]
    #    for k in self.settings.keys():
    #        self.settings[k] = new_settings[k].get()

    def checkIfBoardWasMoved(self, img: B3CImage) -> bool:
        if self.resize:
            img = cv2.resize(img, None, fx=0.25,fy=0.25)
        fgmask = self.fgbg.apply(img, self.settings['MotionDetectionFactor'])
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        self.showDebug(debugkeys.BoardMotion, fgmask)

        if fgmask.sum() > 400:
            return True
        else:
            return False
            #Signals.emit(OnBoardMoved)


class MotionDetectionHandTracker(DebugInfoProvider):
    def __init__(self, img:B3CImage, resize:bool=True) -> None:
        DebugInfoProvider.__init__(self)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands



        self.resize=resize
        if self.resize:
            img = cv2.resize(img, None, fx=0.25,fy=0.25)
        self.imgLast = img
        self.motion_active = False
        for key in debugkeys:
            self.available_debug_info[key.name] = False

    def hasNoMotion(self, img: B3CImage) -> bool:
        if self.resize:
            img = cv2.resize(img, None, fx=0.25,fy=0.25)
        annotated_image = img.copy()

        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4) as hands:
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
        cv2.imshow('hand', annotated_image)
        cv2.waitKey(1)
        self.showDebug(debugkeys.BoardMotion, annotated_image)

        if not self.motion_active and results.multi_hand_landmarks is not None:
            # hand onto of board
            self.motion_active = True
            return False

        if self.motion_active and results.multi_hand_landmarks is None:
            self.motion_active = False
            self.hist = 0
            logging.debug('No Board shift detected')
            return True

        return False

class MotionDetectionBGSubCNT(DebugInfoProvider):
    '''
        Detects Motion between two frames and blocks the classification algorithm
        based on a simple threshold with an histersis to prevent early unocking
    
    '''
    def __init__(self, img: B3CImage, resize:bool = True) -> None:
        DebugInfoProvider.__init__(self)
        
        self.resize=resize
        if self.resize:
            img = cv2.resize(img, None, fx=0.25,fy=0.25)
        self.imgLast = img
        self.hist = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.fgbg = bgsubcnt.createBackgroundSubtractor() 
        self.motion_active = False
        for key in debugkeys:
            self.available_debug_info[key.name] = False

    def hasMotion(self, img: B3CImage) -> bool:
        if self.resize:
            img = cv2.resize(img, None, fx=0.25,fy=0.25)
        img = toByteImage(toGrayImage(img))
        fgmask = self.fgbg.apply(img)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)

        self.showDebug(debugkeys.Motion, fgmask)

        if not self.motion_active and fgmask.sum() > 3060:
            # hand onto of board
            self.motion_active = True
            self.hist = 0
            return True

        if self.motion_active and fgmask.sum() <= 3060:
            #if self.hist < 1:
            #    self.hist += 1
            #    return True
            #else:
            # hand out of board
            self.motion_active = False
            self.hist = 0
            logging.debug('No Board shift detected')
            return False

        return True

