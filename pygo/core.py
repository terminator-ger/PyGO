import cv2
import pdb
import logging

from pygo.classifier import *
from pygo.Motiondetection import MotionDetectionMOG2
from pygo.GoBoard import GoBoard
from pygo.utils.plot import plot_overlay
from pygo.utils.typing import B3CImage, Image, NetMove
from pygo.utils.data import save_training_data
from pygo.utils.misc import flattenList
from pygo.Game import Game
from pygo.Webcam import Webcam

class PyGO:
    def __init__(self):
        self.webcam = Webcam()

        self.img_cam = self.webcam.read()[1]
        self.img_overlay = self.img_cam
        self.Motiondetection = MotionDetectionMOG2(self.img_cam)
        self.Masker = MotionDetectionMOG2(self.img_cam, resize=False)
        self.Board = GoBoard(self.webcam.getCalibration())
        self.Game = Game()
        self.msg = ''
        self.Katrain = None
    
        self.PatchClassifier = CircleClassifier(self.Board, 19)

    def loop10x(self) -> None:
        for _ in range(10):
            self.run_once()

    def loop(self) -> None:
        while (True):
            self.run_once()
    
    def run_once(self) -> None:
            self.img_cam = self.webcam.read()[1]
            self.msg = ''

            if self.Board.hasEstimate:
                self.img_overlay = self.Board.extract(self.img_cam)
                if not self.Motiondetection.hasMotion(self.img_overlay):
                    if self.PatchClassifier.hasWeights:
                        val = self.PatchClassifier.predict(self.img_overlay)
                        logging.debug(val.reshape(19,19))
                        self.msg = self.Game.updateState(val)
                        if self.Katrain is not None:
                            self.Katrain.send(self.msg)
            else:
                self.img_overlay = self.Board.get_corners_overlay(self.img_cam)
                
                # draw possible detected board

            #self.update()
        

