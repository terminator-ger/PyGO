import os
import cv2
import numpy as np

from typing import Optional, Tuple, List

from pygo.Signals import *
from pygo.CameraCalib import CameraCalib

class Webcam:
    def __init__(self, default=None):
        self.cam : Optional[FileVideoStream] = None
        self.camera_calib : Optional[CameraCalib] = None
        self.limit_resolution : Optional[Tuple[int,int]] = (480,640)
        self.scale_factor = 1
        self.default = default
        self.dx = 0
        self.dy = 0
        self.current_port = None
        self.__is_paused = False
        self.__update_ports()
        self.__auto_calibrate()
        Signals.subscribe(GamePause, self.pause_stream)
        Signals.subscribe(GameRun, self.unpause_stream)

    def pause_stream(self, *args):
        self.__is_paused = True

    def unpause_stream(self, *args):
        self.__is_paused = False


    def set_input_file_stream(self, file : str = None) -> None:
        self.cam.release()
        self.cam = cv2.VideoCapture(file)
        self.current_port = file

        width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height= self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if height > self.limit_resolution[0]:
            fy = self.limit_resolution[0] / height #img.shape[0]
            fx = self.limit_resolution[1] / width #img.shape[1]
            self.scale_factor = max(fx, fy)
        else:
            self.scale_factor = 1
        delta = np.array([self.scale_factor*height, self.scale_factor*width]) - np.array(self.limit_resolution)
        self.dx = int(delta[0])
        self.dy = int(delta[1])
        Signals.emit(OnInputChanged)
    
    def __get_next_frame(self) -> np.ndarray:
        img = self.cam.read()[1]
        if self.limit_resolution:
                img_ = cv2.resize(img, dsize=None, 
                                fx = self.scale_factor, 
                                fy = self.scale_factor)

                if self.dx > 0 and self.dy > 0:
                    img_ = img_[self.dx//2 : -self.dx//2, self.dy//2: -self.dy//2]
                elif self.dx == 0 and self.dy > 0:
                    img_ = img_[:, self.dy//2: -self.dy//2]
                elif self.dy == 0 and self.dx > 0:
                    img_ = img_[self.dx//2: -self.dx//2]
                img = img_

        return img

    def read_ignore_lock(self) -> np.ndarray:
        return self.__get_next_frame()

    def read(self) -> np.ndarray:
        if self.__is_paused:
            return None
        else:
            return self.__get_next_frame()


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


    def switch(self) -> None:
        print("Select Camera to use:")
        for i, port in enumerate(self.working_ports):
            print("({}) : {}".format(i, port))

        sel = input()
        if sel.isnumeric:
            self.default_port = self.working_ports[int(sel)]
            print('Switched to {}'.format(self.default_port))

    def getWorkingPorts(self) -> List[str]:
        return [port for port in self.working_ports]


    def __update_ports(self) -> None:
        """
        Test the ports and returns a tuple with the available ports and the ones that are working.
        https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
        """
        self.non_working_ports = []
        dev_port = 0
        self.working_ports = []
        self.available_ports = []
        self.default_port = 0
        best_resolution = 0
        while len(self.non_working_ports) < 5: # if there are more than 5 non working ports stop the testing.
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                self.non_working_ports.append(dev_port)
                logging.debug2("Port %s is not working." %dev_port)
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    self.working_ports.append(dev_port)
                    if w*h > best_resolution:
                        best_resolution = w*h
                        self.default_port = dev_port
                else:
                    logging.debug2("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                    self.available_ports.append(dev_port)
            dev_port +=1

        port = self.default if self.default is not None else self.default_port

        if port is not None:
            print("Using port {}".format(port))
            self.cam = cv2.VideoCapture(port)
            self.current_port = '/dev/video{}'.format(port)
