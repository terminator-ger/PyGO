import cv2
import pdb
import logging

from pygo.classifier import *
from pygo.Motiondetection import *
from pygo.GoBoard import GoBoard
from pygo.utils.plot import Plot
from pygo.utils.debug import Timing
from pygo.Game import Game
from pygo.Webcam import Webcam
from pygo.Signals import *

logging.DEBUG2 = 5
logging.addLevelName(logging.DEBUG2, "DEBUG2")
logging.Logger.debug2 = lambda inst, msg, *args, **kwargs: inst.log(logging.DEBUG2, msg, *args, **kwargs)
logging.debug2 = lambda msg, *args, **kwargs: logging.log(logging.DEBUG2, msg, *args, **kwargs)



class PyGO(Timing):
    def __init__(self):
        Timing.__init__(self)
        self.input_stream = Webcam()

        self.img_cam = self.input_stream.read()[1]
        self.img_overlay = self.img_cam
        self.img_cropped = self.img_cam
        self.img_virtual = self.img_cam

        self.Motiondetection = MotionDetectionMOG2(self.img_cam)
        #self.Motiondetection = MotionDetectionHandTracker(self.img_cam)
        #self.BoardMotionDetecion = BoardShakeDetectionMOG2()
        #self.Motiondetection = MotionDetectionBGSubCNT(self.img_cam)

        self.Board = GoBoard(self.input_stream.getCalibration())
        self.Plot = Plot()
        self.Game = Game()
        self.msg = ''
        self.Katrain = None
        self.input_is_frozen = False
    
        self.PatchClassifier = CircleClassifier(self.Board, 19)

    def freeze(self) -> None:
        self.input_is_frozen = True if not self.input_is_frozen else False

    def startNewGame(self, *args) -> None:
        self.img_cam = self.input_stream.read()
        if not self.Board.hasEstimate:
            self.Board.calib(self.img_cam)
        
        if self.Board.hasEstimate:
            self.Game.startNewGame(19)
            self.img_cropped = self.Board.extract(self.img_cam)
            val = self.PatchClassifier.predict(self.img_cropped)
            self.Game.updateState(val)


    def loop10x(self) -> None:
        for _ in range(10):
            self.run_once()

    def loopDetect10x(self) -> None:
        for _ in range(10):
            img_cam = self.input_stream.read()[1]
        self.Board.calib(img_cam)
        for _ in range(10):
            self.run_once()


    def loop(self) -> None:
        while (True):
            self.run_once()
    
    def run_once(self) -> None:
            if not self.input_is_frozen:
                self.img_cam = self.input_stream.read()
            self.msg = ''

            if self.Board.hasEstimate:
                self.img_cropped =  self.Board.extract(self.img_cam)
                if self.Motiondetection.hasNoMotion(self.img_cropped) and not self.Game.isPaused():
                    i = [x for x in os.listdir('.') if 'out' in x]
                    i = len(i)
                    fn = 'out{}.png'.format(i+1)
                    cv2.imwrite(fn, self.img_cropped)
                    
                    if self.PatchClassifier.hasWeights:
                        #if self.BoardMotionDetecion.checkIfBoardWasMoved(self.img_cropped):
                        #    logging.debug('Realign Board')
                        #    self.Board.calib(self.img_cam)

                        val = self.PatchClassifier.predict(self.img_cropped)
                        self.img_overlay = self.Plot.plot_overlay(val, 
                                                        self.Board.go_board_shifted, 
                                                        self.img_cropped)
                        self.img_virtual = self.Plot.plot_virt_grid(val, 
                                                        self.Board.grd_overlay, 
                                                        self.Board.grid_img)


                        self.Game.updateState(val)
                        #if self.Katrain is not None:
                        #    self.Katrain.send(self.msg)
                else:
                    #overlay old state during motion
                    self.img_overlay = self.Plot.plot_overlay(self.Game.state,
                                                                self.Board.go_board_shifted,
                                                                self.img_cropped)
                    self.img_virtual = self.Plot.plot_virt_grid(self.Game.state, 
                                                        self.Board.grd_overlay, 
                                                        self.Board.grid_img)
            else:
                #self.tic('coverlay')
                img = self.Board.get_corners_overlay(self.img_cam)
                #self.toc('coverlay')
                self.img_overlay = img
                self.img_virtual = img
                self.img_cropped = img
                