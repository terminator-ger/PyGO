import numpy as np
from pygo.Signals import Signals, OnCameraGeometryChanged
import pdb

class CameraCalib:
    def __init__(self, intr) -> None:
        self.fx = 1
        self.fy = 1
        self.scale_mat = np.eye(3)
        self.scale_mat[0,0] = self.fx
        self.scale_mat[1,1] = self.fy

        self.intr = intr
        self.focal = intr[0,0]
        self.center = (intr[0,2], intr[1,2])

        self.calib_size = (640, 480)

    def set_image_size(self, img_shape):
        '''
            img_shape = (width, height)
        '''
        self.fx = img_shape[0] / self.calib_size[0]
        self.fy = img_shape[1] / self.calib_size[1]

        self.scale_mat[0,0] = self.fx
        self.scale_mat[1,1] = self.fy

        Signals.emit(OnCameraGeometryChanged)

    
    def get_intr(self):
        return self.scale_mat @ self.intr
        
    def get_focal(self):
        return self.focal * self.fx

    def get_center(self):
        center = self.center
        return (center[0] * self.fx, center[1] * self.fy)


