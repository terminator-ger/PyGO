import os
import cv2
from typing import Optional, Tuple
import numpy as np
from pygo.CameraCalib import CameraCalib



class Video:
    def __init__(self):
        self.cam : Optional[cv2.VideoCapture] = None
        self.camera_calib : Optional[CameraCalib] = None
        self.__auto_calibrate()


    def read(self) -> Tuple[bool, np.ndarray]:
        return self.cam.read()

    def release(self) -> None:
        self.cam.release()

    def __auto_calibrate(self) -> None:
        #check for existing calibrations
        if os.path.exists('config/calib.npy'):
            self.camera_calib = CameraCalib(np.load('config/calib.npy'))
        else:
            # automatic calibration
            print("Automatic calibration is currently not implemented! - please provide a calibration file - use calib.py")
            self.camera_calib = None
            exit()

    def getCalibration(self) -> CameraCalib:
        return self.camera_calib
