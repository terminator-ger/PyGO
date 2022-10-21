from __future__ import print_function
from __future__ import division
from typing import Tuple, TypeVar, Callable
import matplotlib.pyplot as plt
import cv2 
import pdb
from pygo.utils.image import toCMYKImage, toByteImage, toGrayImage, toColorImage, toHSVImage
import numpy as np
from functools import partial
import math
from pygo.Webcam import Webcam

T = TypeVar('T', int, float)

class Argument:
    name: str
    value: int
    range: Tuple[float,float]
    scaling: Callable
    unscale: Callable
    _dtype: T

    def __init__(self, name, range, value):
        self.name = name
        self.range = range
        self.scaling = lambda x: ((x-range[0]) / (range[1]-range[0]) * 100)
        self.unscale = lambda x: (x/100*(range[1]-range[0])+range[0])
        
        if isinstance(value, int):
            self._dtype = int
        elif isinstance(value, float):
            self._dtype = float

        self.value = int(self.scaling(value))
        print(value)
        print(range)
        print(self.value)
        print(self.unscale(self.value))
    
class CV2ArgTester:
    def __init__(self, img):
        self.args = {}
        self.win_title = 'CV2ArgTester'
        self.win = cv2.namedWindow(self.win_title)
        self.img = img
        self.img_ = None

    def set(self, name: str, value):
        self.args[name].value = value
        #self.args[name].scaling(value)

    def get(self, name: str) -> float:
        return self.args[name]._dtype(self.args[name].unscale(self.args[name].value))

    def addArgument(self, name: str, value, range: Tuple[float,float]):
        self.args[name] = Argument(value, range, value)
        cv2.createTrackbar(name,
                            self.win_title , 
                            self.args[name].value,
                            100,
                            self.update(name))


    def update(self, name, value=None):
        def partial_update(value):
            self.set(name, value)
            self.redraw()
            self.show()
        return partial_update

    def redraw(self):
        raise NotImplementedError()

    def show(self):
        raise NotImplementedError()

class Hough(CV2ArgTester):
    def redraw(self):
       #img = cv2.Canny(toGrayImage(img),50,150,apertureSize = 3)

        _, img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        img_ = img.copy()
        
        lines = cv2.HoughLines(img, 
                                rho=        self.get('rho'), 
                                theta=      self.get('theta'), 
                                threshold=  self.get('threshold'))
                                #srn=        self.get('srn'), 
                                #stn=        self.get('stn')) 
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                # convert both to positive values -> map to first quadrant
                if rho < 0:
                    continue
                a = np.abs(math.cos(theta))
                b = np.abs(math.sin(theta))
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(img_, pt1, pt2, (255,255,255), 1, cv2.LINE_AA)
        self.img_ = img_


    def show(self):
        print('____________________________')
        for k in self.args:
            print('{} : {}'.format(k, self.get(k)))
        if self.img_ is not None:
            cv2.imshow('CV2ArgTester', self.img_)
        else:
            cv2.imshow('CV2ArgTester', self.img)
        cv2.waitKey(0)

from pygo.utils.typing import *
import logging
from nptyping import NDArray
from lu_vp_detect import VPDetection, LS_ALG
from pygo.CameraCalib import CameraCalib

class NoVanishingPointsDetectedException(Exception):
    pass

class BoardDetector(CV2ArgTester):
    def __init__(self, imgs):
        CV2ArgTester.__init__(self, None)
        self.imgs = imgs
        self.img_ = self.stackImgs(self.imgs)
        CameraCalib = Webcam().getCalibration()
        self.vp = VPDetection(focal_length=CameraCalib.focal, 
                              principal_point=CameraCalib.center, 
                              length_thresh=50,
                              line_search_alg=LS_ALG.LSD)
        self.fn = self.segment
    
    def stackImgs(self, imgs):
        imga = np.hstack(imgs[:2])
        imgb = np.hstack(imgs[2:4])
        imgc = np.hstack(imgs[4:])
        img = np.vstack((imga,imgb,imgc))
        img = cv2.resize(img, None, fx=.75, fy=.75)
        return img

    def redraw(self):
        self.img_mask = []
        detected = []
        for img in self.imgs:
            detected.append(self.fn(img))
        self.img_ = self.stackImgs(detected)

    def show(self):
        print('____________________________')
        
        for k in self.args:
            print('{} : {}'.format(k, self.get(k)))
        if self.img_ is not None:
            cv2.imshow('CV2ArgTester', self.img_)
        else:
            cv2.imshow('CV2ArgTester', self.img)
        cv2.waitKey(0)

    def get_vp(self, img: B1CImage) -> Tuple[Point3D, Point3D]:
        van_points = self.vp.find_vps(img)
        if van_points is None:
            raise NoVanishingPointsDetectedException()
        #lines_vp = []
        vps = self.vp.vps_2D
        vp3d = self.vp.vps
        vert_vp = np.argmax(np.abs(vp3d[:,2]))
        vps = np.delete(vps, vert_vp, axis=0)
        vp1 = np.array([vps[0,0], vps[0,1], 1])
        vp2 = np.array([vps[1,0], vps[1,1], 1])
        return vp1, vp2
    

    def get_corners(self, vp1: Point2D, vp2: Point2D, img: Image) -> NDArray:
        '''
        vp1: 2d vanishing point
        vp2: 2d vanishing point
        img: thresholded image of the board
        '''

        contours, hierarchy = cv2.findContours(img, 
                                                cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)
        # vp1 and vp2 are the horizontal and vertical vps
        min_dist = 10000
        corners = []
        corners_mat = None
        #cv2.imshow('',img.copy())
        #cv2.waitKey(1000)

        best_image = np.zeros_like(img)
        img_out = None
        min_area = np.prod(img.shape)
        for c, cnt in enumerate(contours):
            if len(cnt) < 4:
                continue
            img_out = img
            #expect at least half
            area = cv2.contourArea(cnt)
            if area > 0.2*min_area:

                #somethimes the contour has small dents .. approximate till we have four corners left
                # to demonstrate the impact of contour approximation, let's loop
                # over a number of epsilon sizes

                for eps in np.linspace(0.001, 0.05, 10):
                    # approximate the contour
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    # debug output
                    output = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
                    #cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
                    #cv2.imshow("PyGO2", output)
                    #cv2.waitKey(10)

                    #text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
                    #cv2.putText(output, text, (0, 0), cv2.FONT_HERSHEY_SIMPLEX,\
                    #    0.9, (0, 255, 0), 2)
                    ## show the approximated contour image
                    #print("[INFO] {}".format(text))
                    # debug output end

                    if len(approx) == 4:
                        # when we found an approximation with only four corneres we can stop
                        cnt = approx
                        break

                if len(approx) != 4:
                    # in case we looped to the end without a good result goto next shape
                    logging.debug("False corner count, we have : {}".format(len(approx)))
                    continue

                # Find corners on the mask
                mask = np.zeros((img.shape),np.uint8)
                mask = cv2.fillConvexPoly(mask, np.array(approx), 255)
                
                corners = cv2.goodFeaturesToTrack(mask, \
                                                    maxCorners=4, \
                                                    qualityLevel=0.1, \
                                                    minDistance=200)
                if corners is None:
                    continue
                corners = self.sort_corners(corners)
                
                if corners is not None and len(corners) >= 4:
                    topmost = corners[0]
                    rightmost = corners[1]
                    bottommost = corners[2]
                    leftmost = corners[3]
                    corner_mask = np.zeros((img.shape), np.uint8)
                    corner_mask = cv2.fillConvexPoly(corner_mask, corners[:,None,:].astype(int), 255)
                    dev_pixels = np.sum(mask-corner_mask)

                    if dev_pixels > 0.2 * np.sum(mask):
                        #plt.subplot(121)
                        #plt.imshow(mask-corner_mask)
                        #plt.subplot(122)
                        #plt.scatter(corners[:,0],corners[:,1])
                        #plt.imshow(img)
                        #plt.show()
                        #print('skip')
                        continue

                    d1 = min(self.line_point_distance(vp1, leftmost, topmost), 
                            self.line_point_distance(vp2, leftmost, topmost))
                    d2 = min(self.line_point_distance(vp1, rightmost, bottommost),
                            self.line_point_distance(vp2, rightmost, bottommost))

                    d3 = min(self.line_point_distance(vp2, rightmost, topmost),
                            self.line_point_distance(vp1, rightmost, topmost))
                    d4 = min(self.line_point_distance(vp2, leftmost, bottommost),
                            self.line_point_distance(vp1, leftmost, bottommost))

                    # best detection has minimal deviation from vp and least deviation 
                    # from four corners area to poly area

                    dist = d1 + d2 + d3 + d4
                    if dist < min_dist:
                        min_dist = dist
                        #image = cv2.drawContours(img_out, contours, c, (0, 255, 0), 3)
                        #cv2.imshow('',image)
                        #cv2.waitKey(100)

                        #best_image = image
                        #best_cnt = cnt
                        #print(area)
                        corners = [leftmost, topmost, rightmost, bottommost]
                        corners_mat = np.array(corners)
        #if corners_mat is not None:
        #    plt.imshow(best_image)
        #    plt.scatter(corners_mat[:,0], corners_mat[:,1])
        #    plt.show()

        return corners_mat
    
    def detect_board_corners(self, vp1: Point2D, vp2: Point2D, img_bw: B1CImage) -> NDArray:
        #img_bw = cv2.equalizeHist(img_bw)
        corners = self.get_corners(vp1, vp2, img_bw)
        #pdb.set_trace()
        #plt.imshow(img_bw)
        #plt.scatter(corners[:,0], corners[:,1])
        #plt.show()

        if corners is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # loop over the kernels sizes
            for _ in range(3):
                cv2.morphologyEx(src=img_bw, dst=img_bw, op=cv2.MORPH_OPEN, kernel=kernel)
                corners = self.get_corners(vp1, vp2, img_bw)
                #up to threee filter layers
                if corners is not None:
                   break

        logging.debug('Corners {}'.format(corners))
 
        return corners

    def get_corners_overlay(self, img: B3CImage) -> NDArray:
        #img_gray = toGrayImage(img)
        #img_gray = toHSVImage(img)[:,:,1]
        # median blur
        median = cv2.medianBlur(img, 5)

        # threshold on black
        lower = (0,0,0)
        upper = (15,15,15)
        img_gray = cv2.inRange(median, lower, upper)
        #self.tic('get_vp')
        try:
            vp1, vp2 = self.get_vp(img_gray)
        except NoVanishingPointsDetectedException:
            return img
        #self.toc('get_vp')
        #print(vp1, vp2)
        #vpd = self.vp.create_debug_VP_image()
        #cv2.imshow('New', vpd)
        #cv2.waitKey(1)
        thresh, img_bw = cv2.threshold(img_gray, \
                                    0, \
                                    255, \
                                    cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        self.img_mask.append(img_bw)
        #cv2.imshow('overlay', img_bw)
        #cv2.waitKey(1)
        # assumption most lines in the image are from the go board -> vp give us the plane
        # the contour which belongs to those vp is the board
        corners = self.detect_board_corners(vp1, vp2, img_bw.copy())
        img_ = cv2.cvtColor(img_bw.copy(), cv2.COLOR_GRAY2BGR)
        if corners is not None:
            corners = np.int32([corners])
            cv2.polylines(img_, corners, color=(0,255,0), isClosed=True, thickness=3)
        return img_


    def sort_corners(self, corners) -> Optional[NDArray]:
        '''
            corners will be sorted by their center of mass in clockwise orientation
        '''
        if len(corners) > 1:
            center = np.mean(corners,0)
            diff = corners - center
            angles = []
            for vec in diff.squeeze():
                angles.append(np.arctan2(vec[1], vec[0]))
            corners = corners[np.argsort(angles)]
            return corners.squeeze()
        else:
            return None

    def line_point_distance(self, point, line_s, line_e):
        '''
        '''
        nom = np.abs((line_e[0]-line_s[0])*(line_s[1]-point[1]) - (line_s[0]-point[0])*(line_e[1]-line_s[1]))
        denom = np.sqrt((line_e[0]-line_s[0])**2 + (line_e[1]-line_s[1])**2) + 1e-12
        return nom/denom

    def sort_corners(self, corners) -> Optional[NDArray]:
        '''
            corners will be sorted by their center of mass in clockwise orientation
        '''
        if len(corners) > 1:
            center = np.mean(corners,0)
            diff = corners - center
            angles = []
            for vec in diff.squeeze():
                angles.append(np.arctan2(vec[1], vec[0]))
            corners = corners[np.argsort(angles)]
            return corners.squeeze()
        else:
            return None

    
    def segment(self, img):
        # median blur
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #median = cv2.medianBlur(img, 5)
        img = cv2.equalizeHist(img)

        # threshold on black
        l = self.get('lower')
        u = self.get('upper')
        lower = (l)
        upper = (u)
        #img = cv2.inRange(img, lower, upper)
        thresh, img = cv2.threshold(img, \
                                    0, \
                                    50, \
                                    cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return img

if __name__ == '__main__':
    imgs = [cv2.imread('light0{}.png'.format(i)) for i in range(1,7)]
    argtest = BoardDetector(imgs)
    argtest.addArgument('lower',0,(0,100))
    argtest.addArgument('upper',15,(0,100))

 
    argtest.show()

