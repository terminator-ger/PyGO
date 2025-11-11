import cv2
#import lsd
import numpy as np
import matplotlib.pyplot as plt
from pygo.GoBoard import GoBoard, CORNER_DETECTION_ALG
cv2.__version__


cap = cv2.VideoCapture("E:\\dev\\pygo_data\\AlphaGo (W) vs Ke Jie (B) $1.5 million game with AlphaGo, relaxing game of Go on a real board [GhVDiAjN-h4].mp4")
def read_next(cap):
    for i in range(10):
        ret, img = cap.read()
    return img
board = GoBoard(camera_calibration=None, corner_detection_alg=CORNER_DETECTION_ALG.CPD)
_, img0 = cap.read()
board.calib(img0)
img = board.extract(img0)
plt.imshow(img)
plt.show()