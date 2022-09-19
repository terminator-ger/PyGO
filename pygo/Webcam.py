import os
import cv2
from typing import Optional, Tuple
import numpy as np
from CameraCalib import CameraCalib



class Webcam:
    def __init__(self):
        self.cam : Optional[cv2.VideoCapture] = None
        self.camera_calib : Optional[CameraCalib] = None
        self.__update_ports()
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


    def switch(self) -> None:
        print("Select Camera to use:")
        for i, port in enumerate(self.working_ports):
            print("({}) : {}".format(i, port))
        
        sel = input()
        if sel.isnumeric:
            self.default_port = self.working_ports[int(sel)]
            print('Switched to {}'.format(self.default_port))
        


    def __update_ports(self) -> None:
        """
        Test the ports and returns a tuple with the available ports and the ones that are working.
        https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
        """
        self.non_working_ports = []
        dev_port = 0
        self.working_ports = []
        self.available_ports = []
        self.default_port = None
        best_resolution = 0
        while len(self.non_working_ports) < 1: # if there are more than 5 non working ports stop the testing. 
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                self.non_working_ports.append(dev_port)
                #print("Port %s is not working." %dev_port)
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    self.working_ports.append(dev_port)
                    if w*h > best_resolution:
                        best_resolution = w*h
                        self.default_port = dev_port
                        print(dev_port)
                else:
                    #print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                    self.available_ports.append(dev_port)
            dev_port +=1

        if self.default_port is not None:
            print("Using port {}".format(self.default_port))
            self.cam = cv2.VideoCapture(self.default_port)
