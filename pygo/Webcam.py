import os
import numpy as np
import cv2


class CameraCalib:
    def __init__(self, intr) -> None:
        self.focal = intr[0,0]
        self.intr = intr
        self.center = (intr[0,2], intr[1,2])


class Webcam:
    def __init__(self):
        self.__update_ports()
        self.__auto_calibrate()

        self.cam = cv2.VideoCapture(self.default_port)
        self.camera_calib = None

    def read(self):
        return self.cam.read()
        
    def __auto_calibrate(self):
        #check for existing calibrations
        if os.path.exists('../calib.npy'):
            self.camera_calib = CameraCalib(np.load('../calib.npy'))
        else:
            # automatic calibration
            print("Automatic calibration is currently not implemented! - please provide a calibration file - use calib.py")
            exit()
    
    def getCalibration(self):
        return self.camera_calib


    def switch(self):
        print("Select Camera to use:")
        for i, port in enumerate(self.working_ports):
            print("({}) : {}".format(i, port))
        
        sel = input()
        if sel.isnumeric:
            self.default_port = self.working_ports[int(sel)]
            print('Switched to {}'.format(self.default_port))
        


    def __update_ports(self):
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
        while len(self.non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                self.non_working_ports.append(dev_port)
                #print("Port %s is not working." %dev_port)
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    #print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                    self.working_ports.append(dev_port)
                    if w*h > best_resolution:
                        best_resolution = w*h
                        self.default_port = dev_port
                else:
                    #print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                    self.available_ports.append(dev_port)
            dev_port +=1
