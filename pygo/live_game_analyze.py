from __future__ import division
from ctypes import resize
import enum
import cv2
import pdb
import sys
import math

import numpy as np

from multiprocessing.connection import Client

from classifier import GoClassifier, HaarClassifier, IlluminanceClassifier, CircleClassifier
from Motiondetection import MotionDetectionMOG2
from GoBoard import GoBoard
from utils.data import save_training_data
from utils.misc import flattenList
from utils.plot import plot_overlay
from utils.debug import DebugInfo
from Game import Game, GameState
from Ensemble import SoftVoting, MajorityVoting
from Webcam import Webcam

from pyELSD.pyELSD import PyELSD
from utils.image import *
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './train')
import warnings
warnings.filterwarnings('always') 



if __name__ == '__main__':
    PLOT_CIRCLES=False
    PLOT_LINES=False

    webcam = Webcam()

    _, img = webcam.read()

    MD = MotionDetectionMOG2(img)
    MASKER = MotionDetectionMOG2(img, resize=False)
    BOARD = GoBoard(webcam.getCalibration())
    GS = Game()
    KATRAIN = None
    
    PatchClassifier = [CircleClassifier(BOARD)]
    #PatchClassifier.append(GoClassifier())
    #PatchClassifier.append(HaarClassifier())
    #PatchClassifier.append(IlluminanceClassifier())
    EnsampleMethod = MajorityVoting(PatchClassifier)

    print('PyGO - Visual Interface for KaTrain')
    print('(c)alibrate      (t)rain')
    print('(n)ew game       (f)nish game')
    print('(d)ebug          (g)enerate training data')
    print('(a)nalyze')

    while True:
        _, img = webcam.read()

        last_key = cv2.waitKey(1) 
        if last_key == ord('c'):
            print('Calibration')
            BOARD.calib(img)
            img = BOARD.extract(img)
      
        elif last_key == ord('t'):
            print("Train classifier")
            PatchClassifier.train()
            PatchClassifier.store()

        elif last_key == ord('g'):
            print('Generate training data \n Place stones in training pattern - Press (c)ontinue')
            patches = []
            while True:
                ret, img = webcam.read()
                last_key = cv2.waitKey(1) 

                if not BOARD.hasEstimate:
                    print("Calibrate the Board first!")
                    break

                if last_key == ord('c'):
                    patches = BOARD.extractOnPoints(img)
                    save_training_data(patches)
                    break
        
        elif last_key == ord('n'):
            print('New Game started')
            GS.startNewGame(19)

        elif last_key == ord('a'):
            print('Live Gama analysis')
            addr = input('Server Address: default[127.0.0.1:8888]') or 'localhost:8888'
            addr = addr.split(':')
            if len(addr) == 1:
                # add default port
                addr.append('8888')
            net_addr = (addr[0], int(addr[1]))
            KATRAIN = Client(net_addr, authkey=b'katrain')
            print('Connected to Katrain')

        elif last_key == ord('f'):
            print('Game finished')
            GS.endGame()
            if KATRAIN is not None:
                KATRAIN.close()

        elif last_key == ord('q'):
            print('Good Bye!')
            if GS.GS == GameState.RUNNING:
                GS.endGame()
            if KATRAIN is not None:
                KATRAIN.close()
            break
        elif last_key == ord('d'):
            debug = DebugInfo([MD, MASKER, BOARD, GS, PatchClassifier[0]])
            debug.showOptions()
        
        if BOARD.hasEstimate:
            img = BOARD.extract(img)
            #cv2.imwrite('out.png', img)
            if not MD.hasMotion(img):
                #mask = cv2.pyrMeanShiftFiltering(img, 5, 80, 3)
                #plt.imshow(mask)
                #plt.show()

                #for i in range(5):
                #    mask = MASKER.getMask(img)
                #    plt.imshow(mask)
                #    plt.show(block=False)
                #    ret, img = webcam.read()
                #    pdb.set_trace()
                if PatchClassifier[0].hasWeights:
                    #if last_move is not None:
                    #patches = BOARD.imgToPatches(img)
                    #val = EnsampleMethod.predict(patches)
                    val = PatchClassifier[0].predict(img)
                    print(val.reshape(19,19))
                    msg = GS.updateState(val)
                    if KATRAIN is not None:
                        KATRAIN.send(msg)
                    #last_move = img


            img = plot_overlay(GS.state, BOARD.go_board_shifted, img)
        cv2.imshow('PyGO',img)
        cv2.waitKey(1)

        last_img = img
   
# When everything done, release the capture
webcam.release()
cv2.destroyAllWindows()
