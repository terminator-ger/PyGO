import cv2
import pdb
import logging

from pygo.classifier import *
from pygo.Motiondetection import MotionDetectionMOG2, MotionDetectionBorderMask, MotionDetectionBGSubCNT
from pygo.GoBoard import GoBoard
from pygo.utils.plot import Plot
from pygo.utils.debug import Timing
from pygo.Game import Game
from pygo.Webcam import Webcam

class PyGO(Timing):
    def __init__(self):
        Timing.__init__(self)
        self.webcam = Webcam()

        self.img_cam = self.webcam.read()[1]
        self.img_overlay = self.img_cam
        self.img_cropped = self.img_cam
        self.img_virtual = self.img_cam

        self.Motiondetection = MotionDetectionMOG2(self.img_cam)
        #self.Motiondetection = MotionDetectionBGSubCNT(self.img_cam)
       # self.Motiondetection = MotionDetectionBorderMask()
        #self.Masker = MotionDetectionMOG2(self.img_cam, resize=False)
        self.Board = GoBoard(self.webcam.getCalibration())
        self.Plot = Plot()
        self.Game = Game()
        self.msg = ''
        self.Katrain = None
    
        self.PatchClassifier = CircleClassifier(self.Board, 19)

    def loop10x(self) -> None:
        for _ in range(10):
            self.run_once()

    def loopDetect10x(self) -> None:
        for _ in range(10):
            img_cam = self.webcam.read()[1]
        self.Board.calib(img_cam)
        for _ in range(10):
            self.run_once()


    def loop(self) -> None:
        while (True):
            self.run_once()
    
    def run_once(self) -> None:
            self.img_cam = self.webcam.read()[1]
            self.msg = ''

            if self.Board.hasEstimate:
                #self.tic('extract')
                self.img_cropped = self.Board.extract(self.img_cam)
                #self.toc('extract')
                self.tic('motion')
                if not self.Motiondetection.hasMotion(self.img_cropped):
                    self.toc('motion')

                    if self.PatchClassifier.hasWeights:
                        cv2.imwrite('out.png', self.img_cropped)
                        self.tic('classifier')
                        val = self.PatchClassifier.predict(self.img_cropped)
                        self.toc('classifier')
                        self.tic('overlay1')
                        self.img_overlay = self.Plot.plot_overlay(val, 
                                                        self.Board.go_board_shifted, 
                                                        self.img_cropped)
                        self.toc('overlay1')
                        self.tic('overlay2')
                        self.img_virtual = self.Plot.plot_virt_grid(val, 
                                                        self.Board.grd_overlay, 
                                                        self.Board.grid_img)
                        self.toc('overlay2')

                        logging.debug(val.reshape(19,19))

                        self.tic('gamestate')
                        self.msg = self.Game.updateState(val)
                        self.toc('gamestate')
                        if self.Katrain is not None:
                            self.Katrain.send(self.msg)
            else:
                self.tic('coverlay')
                img = self.Board.get_corners_overlay(self.img_cam)
                self.toc('coverlay')
                self.img_overlay = img
                self.img_virtual = img
                self.img_cropped = img
                