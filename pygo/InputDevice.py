import os
import cv2
import numpy as np
from tkinter import W
from typing import Optional, Tuple, List

from pygo.Signals import *
from pygo.CameraCalib import CameraCalib

class InputDevice:
    cam : Optional[cv2.VideoCapture] = None
    camera_calib : Optional[CameraCalib] = None
    limit_resolution : Optional[Tuple[int,int]] = (480,640)
    scale_factor = 1
    dx = 0
    dy = 0
    current_port = None
    __is_paused = False
    last_frame = None
    frames_total = None
    frame_n = None
    fps = 1
    is_video = False
    current = 0
    default_port = 0
    default = 0

    def __init__(self, default = None, file: str = None):
        self.default = default
        if file is not None:
            self.set_input_file_stream(file)
        else:
            self.__update_ports()

        self.__auto_calibrate()

        CoreSignals().subscribe(GamePause, self.pause_stream)
        CoreSignals().subscribe(GameRun, self.unpause_stream)
        CoreSignals().subscribe(InputStreamSeek, self.set_pos)

        CoreSignals().subscribe(InputForward, self._forward)
        CoreSignals().subscribe(InputForward10, self._forward10)
        CoreSignals().subscribe(InputBackward, self._backward)
        CoreSignals().subscribe(InputBackward10, self._backward10)

    def _forward(self, args):
        cur = self.get_pos()  / self.fps
        next = cur + 60 
        self.frame_n = min(next, self.frames_total)
        self._set_pos(self.frame_n)
        CoreSignals().emit(PreviewNextFrame)


    def _forward10(self, args):
        cur = self.get_pos()  / self.fps
        next = cur + 10 
        self.frame_n = min(next, self.frames_total)
        self._set_pos(self.frame_n)
        CoreSignals().emit(PreviewNextFrame)


    def _backward(self, args):
        cur = self.get_pos()  / self.fps
        next = cur - 60
        self.frame_n = max(next, 0)
        self._set_pos(self.frame_n)
        CoreSignals().emit(PreviewNextFrame)


    def _backward10(self, args):
        cur = self.get_pos()  / self.fps
        next = cur - 10
        self.frame_n = max(next, 0)
        self._set_pos(self.frame_n)
        CoreSignals().emit(PreviewNextFrame)


    def pause_stream(self, *args):
        self.__is_paused = True


    def unpause_stream(self, *args):
        self.__is_paused = False


    def get_length(self) -> Optional[int]:
        return self.frames_total


    def get_pos(self) -> Optional[int]:
        fp = self.cam.get(cv2.CAP_PROP_POS_FRAMES)
        if fp == -1:
            return None
        return fp


    def get_time(self) -> Optional[float]:
        '''
            returns time in with seconds as base
            5.2 -> 5 sec 200 ms
        '''
        f = self.cam.get(cv2.CAP_PROP_POS_FRAMES)
        return f / self.fps


    def set_pos(self, args) -> None:
        '''
            set the time in floating second format
        '''
        self._set_pos(args[0])

    
    def _set_pos(self, time: int) -> None:
        frame = int(time * self.fps)
        logging.info("Video set to {}".format(frame/self.fps))
        max_len = self.get_length()
        if max_len is not None and frame >= 0 and frame < max_len:
            self.cam.set(cv2.CAP_PROP_POS_FRAMES, frame)


    def set_input_file_stream(self, file : str = None) -> None:
        if self.cam is not None:
            self.cam.release()
        self.cam = cv2.VideoCapture(file)
        self.current_port = file
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)

        self.frames_total = self.cam.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.frames_total == -1:
            # no video file
            self.frame_n = 0
            self.is_video = False
        else:
            self.is_video = True

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
        CoreSignals().emit(OnInputChanged)
    
        
    def __get_next_frame(self) -> np.ndarray:
        ret, img = self.cam.read()
        if not ret:
            return self.last_frame
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
            #img = cv2.fastNlMeansDenoising(img, 
            #                            templateWindowSize=5, 
            #                            searchWindowSize=7)
            #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            #img = cv2.filter2D(img, -1, kernel)
            self.last_frame = img
            # only increment for live videos

        return img

    def read_ignore_lock(self) -> np.ndarray:
        return self.__get_next_frame()

    def read(self) -> np.ndarray:
        if self.__is_paused:
            return None
        else:
            frame = self.__get_next_frame()
            # notify ui of video progress
            if self.frames_total == -1:
                self.frame_n += 1
            else:
                UISignals.emit(UIVideoFrameCounterUpdated, self.get_time())
                CoreSignals.emit(VideoFrameCounterUpdated, self.get_time())
            return frame
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.cam is None:
            logging.warning("Input Device not initialized")
            raise StopIteration
        
        ret, frame = self.cam.read()
        if ret:
            return frame
        raise StopIteration


    def release(self) -> None:
        self.cam.release()

    def __auto_calibrate(self) -> None:
        #check for existing calibrations
        if os.path.exists('config/calib.npy'):
            self.camera_calib = CameraCalib(np.load('config/calib.npy'))
            return
        elif self.cam.isOpened():
            # automatic calibration
            # estimate based upon image size
            ret, frame = self.cam.read()
            if not ret:
                logging.error("Automatic calibration is currently not implemented! - please provide a calibration file - use calib.py")
                self.camera_calib = None
                exit()
               
            h, w = frame.shape[0], frame.shape[1]
            calib = np.eye(3, dtype=np.float32)
            calib[0,2] = w//2
            calib[1,2] = h//2
            calib[0,0] = 600
            calib[1,1] = 600
            self.camera_calib = CameraCalib(calib)
            return

        # default fail
        logging.error("Automatic calibration is currently not implemented! - please provide a calibration file - use calib.py")
        self.camera_calib = None
        exit()

    def getCalibration(self) -> CameraCalib:
        return self.camera_calib


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
                logging.debug("Port %s is not working." %dev_port)
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
                    logging.debug("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                    self.available_ports.append(dev_port)
            dev_port +=1

        port = self.default if self.default is not None else self.default_port

        if port is not None:
            logging.info("Using port {}".format(port))
            self.cam = cv2.VideoCapture(port)
            self.current_port = '/dev/video{}'.format(port)
            self.frames_total = -1
            self.frame_n = 0
