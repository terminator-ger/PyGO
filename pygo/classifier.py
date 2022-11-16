import cv2
import warnings

import logging
import numpy as np


from enum import Enum, auto
from typing import Tuple
from dataclasses import dataclass
from scipy.ndimage import label

from skimage.filters import sobel
from skimage.draw import circle_perimeter
from skimage.transform import hough_circle, hough_circle_peaks

from pygo.Signals import OnBoardGridSizeKnown, Signals
from pygo.utils.line import point_in_circle
from pygo.utils.debug import Timing
from pygo.utils.image import *
from pygo.utils.debug import DebugInfoProvider
from pygo.utils.typing import B1CImage, B3CImage, GoBoardClassification
from pygo.GoBoard import GoBoard


warnings.filterwarnings('always') 
def toNP(x):
    return x.detach().cpu().numpy()


class Classifier:
    def predict(self, patches):
        raise NotImplementedError()

    def train(self, patches):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def store(self):
        raise NotImplementedError()


class debugkeys(Enum):
    Mask_Black = auto()
    Mask_White = auto()
    Detected_Intensities = auto()
    IMG_B = auto()
    IMG_W = auto()
    MASK = auto()
    GRID = auto()
    BIN = auto()
    CIRCLE = auto()
    DETECT = auto()


@dataclass
class CV2PlotSettings:
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fontScale : float      = 0.8
    fontColor : Tuple[int] = (255,255,255)
    thickness : int        = 1
    lineType  : int        = 2


class CircleClassifier(Classifier, DebugInfoProvider, Timing):
    def __init__(self, BOARD:GoBoard, size: int) -> None:
        Classifier.__init__(self)
        Timing.__init__(self)
        DebugInfoProvider.__init__(self)

        self.size = size 
        self.hasWeights = True
        self.BOARD=BOARD
        self.params = cv2.SimpleBlobDetector_Params()

        self.params.minThreshold = 127
        self.params.maxThreshold = 255

        self.params.filterByCircularity = True
        self.params.minCircularity = 0.70
        self.params.maxCircularity = 1.0
        
        self.params.filterByConvexity = False
        self.params.minConvexity = 0.0
        self.params.maxConvexity = 1.0

        self.params.filterByArea = True
        self.params.minArea = 50
        self.params.maxArea = 999

        self.params.filterByInertia = False
        self.params.minInertiaRatio = 0.1
        self.params.maxInertiaRatio = 1.0

        self.ks = 35
        self.c = 1
        self.img_debug = None

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            self.detector = cv2.SimpleBlobDetector(self.params)
        else :
            self.detector = cv2.SimpleBlobDetector_create(self.params)


        for key in debugkeys:        
            self.available_debug_info[key.name] = False

        self.cv_settings =  CV2PlotSettings()
        Signals.subscribe(OnBoardGridSizeKnown, self.update_grid_size)



    def predict(self, img: B3CImage) -> GoBoardClassification:
        value = toHSVImage(img)[:,:,2]

        detections_circle = self._detect_circle_detection_on_gradient(img.copy())
        detections_hidden = self._detect_hidden_intersection(img.copy())
        detections_blobb  = self._detect_blobb(img.copy())

        cell_w = (np.mean(np.diff(self.BOARD.go_board_shifted.reshape(19,19,2)[:,:,0], axis=0))//2).astype(int)

        # initial markers
        markers = np.ones_like(map).astype(np.int32)
        mask = np.ones_like(map).astype(np.uint8)
        id = 2
        detected_circles = []
        img_detect = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .7
        color = (255, 0, 0)
        color_detect = (0, 255, 0)
        thickness = 2
        for coord in self.BOARD.go_board_shifted.astype(int):
            isin_circle = np.all(np.equal(coord, detections_circle),axis=1).any() if len(detections_circle) > 0 else False
            isin_hidden = np.all(np.equal(coord, detections_hidden),axis=1).any() if len(detections_hidden) > 0 else False
            isin_blobb  = np.all(np.equal(coord, detections_blobb),axis=1).any() if len(detections_blobb) > 0 else False
            detect_count = np.sum([isin_circle, isin_hidden, isin_blobb])
  
            cv2.putText(img=img_detect, 
                        text=str(detect_count), 
                        org=coord+np.array([-5,5]), 
                        fontFace=font,
                        fontScale=fontScale, 
                        color=color if detect_count <2 else color_detect, 
                        thickness=thickness, 
                        lineType=cv2.LINE_AA)
            if detect_count >= 2:
                cv2.circle(markers, coord, cell_w+3, 0, -1)
                cv2.circle(markers, coord, 2, id, 2)

                cv2.circle(mask, coord, cell_w, 255, -1)
                crl = np.array([coord[0], coord[1], cell_w])
                detected_circles.append([crl])
                id += 1

        self.showDebug(debugkeys.DETECT, img_detect)
        self.showDebug(debugkeys.MASK, mask)

        val,_ = self.__analyse(detected_circles, value)
        
        # our coordinate system is rotated
        val = val.reshape(19,19).astype(int)

        return val


    def update_grid_size(self, args):
        grid = args[0].reshape(19,19,2)
        dx = np.mean(np.diff(grid[:,:,0].T))
        dy = np.mean(np.diff(grid[:,:,1]))
        a = ((np.mean([dx,dy])/2)**2)*np.pi
        self.params.minArea = a *0.7
        self.params.maxArea = a *1.3
  

    def segment_on_dt(self, a, img):
        border = cv2.dilate(img, None, iterations=1)
        border = border - cv2.erode(border, None)

        dt = cv2.distanceTransform(img, 2, 3)
        cv2.normalize(dt, dt, 0, 1.0, cv2.NORM_MINMAX)
        #dt_color = cv2.applyColorMap(toByteImage(dt), cv2.COLORMAP_JET)
        #cv2.imshow('asd', dt_color)

        _, dt = cv2.threshold(dt, 0.8, 1.0, cv2.THRESH_BINARY)
        #cv2.waitKey()

        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
        # Completing the markers now. 
        lbl[border == 255] = 255
        #cv2.imshow('label', lbl)

        lbl = lbl.astype(np.int32)
        cv2.watershed(a, lbl)

        lbl[lbl == -1] = 0
        lbl = lbl.astype(np.uint8)
        return 255 - lbl

    def remove_glare2(self, img):
        GLARE_MIN = np.array([0, 0, 50],np.uint8)
        GLARE_MAX = np.array([0, 0, 225],np.uint8)

        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        #HSV
        frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)
        
        #INPAINT + HSV
        result = cv2.inpaint(img, frame_threshed, 0.1, cv2.INPAINT_TELEA) 

        #HSV+ INPAINT + CLAHE
        lab1 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        lab_planes1 = cv2.split(lab1)
        clahe1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        lab_planes1[0] = clahe1.apply(lab_planes1[0])
        lab1 = cv2.merge(lab_planes1)
        clahe_bgr1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)
        return clahe_bgr1

    def create_mask_white(self, ipt):
        mask_white = cv2.equalizeHist(ipt)
        mask_white[mask_white>25] = 255
        mask_white[mask_white<15] = 0
        mask_white =  cv2.adaptiveThreshold(mask_white, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 
                                        31, 1)
        return mask_white

    def binarize(self, img: B1CImage) -> B1CImage:
        gray = toByteImage(img.copy())
        edge = toByteImage(sobel(img))
        
        edge = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        gray = self.remove_lines(gray, edge, 'h')
        gray = self.remove_lines(gray, edge, 'v')
        gray = cv2.GaussianBlur(gray,(5,5),0)
        #cv2.imshow('gray', gray)
        grid = toByteImage(sobel(gray))
        out = cv2.threshold(grid, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # remove edges outside the boards zone which could interfere with the detection
        bs = self.BOARD.border_size+3
        cw = int(self.BOARD.cell_w/2)
        ch = int(self.BOARD.cell_h/2)
        h,w = img.shape
        out[:, :(bs-cw)] = 0
        out[:,(w-bs+cw):] = 0
        out[:(bs-ch), :] = 0
        out[(h-bs+ch):,:] = 0

        return out


    def remove_lines(self, image, thresh, orientation='h'):
        ipt = image.copy()
        if orientation == 'h':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
            repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6,1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
            repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))

        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(ipt, [c], -1, 255, 1 )

        # Repair image
        result = 255 - cv2.morphologyEx(255 - ipt, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
        return result    


    def remove_circles_not_on_grid(self, circles, grid):
        circles_out = []
        for crl in circles:
            if np.any(point_in_circle(grid, np.array(crl))):
                circles_out.append(crl)
        return circles_out


    def detect_stones(self, binary_img):
        cell_w = (np.mean(np.diff(self.BOARD.go_board_shifted.reshape(19,19,2)[:,:,0], axis=0))//2).astype(int)
        cell_h = (np.mean(np.diff(self.BOARD.go_board_shifted.reshape(19,19,2)[:,:,1], axis=1))//2).astype(int)

        minRadius = min(cell_w, cell_h) -1
        maxRadius = max(cell_w, cell_h) +1

        hough_radii = np.arange(minRadius, maxRadius, 1)
        hough_res = hough_circle(binary_img, hough_radii)
        # Select the most prominent 3 circles
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 
                                                min_xdistance=minRadius,
                                                min_ydistance=minRadius)
        circles = zip(cx,cy,radii)
        circles = self.remove_small_circles(circles, minRadius)
        circles_clean = self.remove_circles_not_on_grid(circles, 
                                                    self.BOARD.go_board_shifted)
        return circles_clean


    def remove_small_circles(self, circles, thresh):
        circles_clean = []
        for (cx,cy,radius) in circles:
            if radius >= thresh:
                circles_clean.append((cx,cy, radius))
        return circles_clean


    def get_hidden_intersections(self, img: B1CImage, scale: float = 1.0) -> B1CImage:
        map = sobel(img)
        map = toByteImage(map)
        cell_w = (np.mean(np.diff(self.BOARD.go_board_shifted.reshape(19,19,2)[:,:,0], axis=0))//2).astype(int)
        cell_w = (int(cell_w*scale))

        if cell_w % 2 == 0:
            cell_w -= 1
        corner = np.zeros((cell_w, cell_w))
        corner[cell_w//2,:cell_w//2] = 255
        corner[:cell_w//2,cell_w//2] = 255
        corner[cell_w//2,cell_w//2] = 255

        side = np.zeros((cell_w, cell_w))
        side[:,cell_w//2] = 255
        side[cell_w//2,:cell_w//2] = 255
        corner = corner.astype(np.uint8)
        side = side.astype(np.uint8)


        ksize = int(cell_w / 2) + 2
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        map_corner = np.zeros_like(map).astype(float)
        factor = np.zeros_like(map).astype(np.int32)
        coord = self.BOARD.go_board_shifted.astype(int).reshape(19,19,-1)
        coord = (coord * scale).astype(int)

        radius = int(cell_w / 2)
        for x in range(19):
            for y in range(19):
                crd = coord[x,y]
                if x in [0,18] or y in [0,18]:
                    cv2.circle(factor, crd, radius, 2, -1)        
                elif np.isin([x,y], [[0,0], [0,18], [18,0], [18,18]]).any():            
                    cv2.circle(factor, crd, radius, 4, -1)        
                else:        
                    cv2.circle(factor, crd, radius, 1, -1)

        for _ in range(4):    
            flt = cv2.morphologyEx(map, cv2.MORPH_ERODE, corner).astype(float)
            flt = (flt - flt.min()) / (flt.max() - flt.min())
            map_corner += flt
            corner = np.rot90(corner)

        map_corner = factor * map_corner
        map_corner = (map_corner - map_corner.min()) / (map_corner.max() - map_corner.min()) * 255
        grid = toByteImage(map_corner)

        _, grid = cv2.threshold(grid, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        grid = cv2.dilate(grid, ellipse, iterations=1)

        return grid


    def _detect_hidden_intersection(self, img):
        clahe = cv2.createCLAHE(clipLimit=2)
        img = toColorImage(clahe.apply(toGrayImage(img)))
        cmyk = toCMYKImage(img)
        hidden_corners = self.get_hidden_intersections(cmyk[:,:,3])
        self.showDebug(debugkeys.GRID, hidden_corners)
        detections = []
        #for coord in self.BOARD.go_board_shifted.astype(int):
        #    if hidden_corners[coord[1], coord[0]] == 0:
        #        detections.append(coord)

        cell_w = (np.mean(np.diff(self.BOARD.go_board_shifted.reshape(19,19,2)[:,:,0], axis=0))//2).astype(int)
        cell_w = (int(cell_w/2))
        T = cell_w*cell_w*0.6
        grid = self.BOARD.go_board_shifted.astype(int)
        for coord in grid:
            if np.sum(hidden_corners[coord[1]-cell_w:coord[1]+cell_w, 
                                     coord[0]-cell_w:coord[0]+cell_w]) / 255 < T:
                detections.append(coord)

        detections = np.array(detections)
        return detections

    def get_hidden_intersection_count(self, img: B3CImage, scale: float = 1.0) -> int:
        '''
        External method, used in the motion detector for cases where someone rests
        with their hands long over the board while thinking...
        '''
        clahe = cv2.createCLAHE(clipLimit=2)
        img = toColorImage(clahe.apply(toGrayImage(img)))
        cmyk = toCMYKImage(img)
        hidden_corners = self.get_hidden_intersections(cmyk[:,:,3], scale)
        scaled_grid = (self.BOARD.go_board_shifted * scale).astype(int)
        hidden_intersections = 0

        cell_w = (np.mean(np.diff(self.BOARD.go_board_shifted.reshape(19,19,2)[:,:,0], axis=0))//2).astype(int)
        cell_w = (int(cell_w*scale/2))
        T = cell_w*cell_w*0.6
        for coord in scaled_grid:
            if np.sum(hidden_corners[coord[1]-cell_w:coord[1]+cell_w, 
                                     coord[0]-cell_w:coord[0]+cell_w]) / 255 < T:
                hidden_intersections += 1

        return hidden_intersections



    def _detect_circle_detection_on_gradient(self, img):
        clahe = cv2.createCLAHE(clipLimit=5)

        img = toColorImage(clahe.apply(toGrayImage(img)))

        cmyk = toCMYKImage(img)
        bin_img = self.binarize(cmyk[:,:,3])
        circles = self.detect_stones(bin_img)

        def plot_circles(image, circles): 
            crcl = toColorImage(image)
            for x,y,r in circles:
                circy, circx = circle_perimeter(y, x, r,
                                                shape=image.shape)
                crcl[circy, circx] = (20, 220, 20)
            return crcl

        crcl = plot_circles(bin_img.copy(), circles)

        self.showDebug(debugkeys.BIN, bin_img)
        self.showDebug(debugkeys.CIRCLE, crcl)
        detections = []
        for coord in self.BOARD.go_board_shifted.astype(int):
            if np.any(point_in_circle(coord, np.array(circles))):
                detections.append(coord)

        detections = np.array(detections)
        return detections


    def _detect_blobb(self, img: B3CImage) -> GoBoardClassification:
        img = self.remove_glare2(img)
        imgb   = toHSVImage(img)[:,:,2]
        imgw  = toHSVImage(img)[:,:,1]
        value = toHSVImage(img)[:,:,2]
        imgb   = toByteImage(imgb)
        imgw  = toByteImage(imgw)       
        self.img_debug = img 
        img_b = cv2.adaptiveThreshold(imgb, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 
                                35, -2)
        
        img_w = cv2.adaptiveThreshold(imgw, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 
                                35, 2)

        self.showDebug(debugkeys.IMG_B, img_b)
        self.showDebug(debugkeys.IMG_W, img_w)

        #self.tic('pred- water')
        mask_b = 255-self.__watershed(255-img_b)
        mask_w = 255-self.__watershed(255-img_w)

        img_masks = [mask_b, mask_w]

        detected_circles = []
        for img in img_masks:
            keypoints = self.detector.detect(img)
            logging.debug('Keypoints found: {}'.format(len(keypoints)))
            if self.debugStatus(debugkeys.Detected_Intensities):
                self.img_debug = cv2.drawKeypoints(self.img_debug, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            for kp in keypoints:
                crl = np.array([kp.pt[0], kp.pt[1], kp.size])
                detected_circles.append([crl])

        self.showDebug(debugkeys.Mask_Black, mask_b)
        self.showDebug(debugkeys.Mask_White, mask_w)
 
        val, det = self.__analyse(detected_circles, value)

        return det


    def __watershed(self, fg : B1CImage) -> B1CImage:
        #fg = toByteImage(fg)
        dist_transform = cv2.distanceTransform(fg,cv2.DIST_L2,3)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        unknown = fg - sure_fg
        # Marker labelling
        ret, markers = cv2.connectedComponents(toByteImage(sure_fg))
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        markers[unknown.astype(bool)]  = 0
        #self.tic('ws')
        markers = cv2.watershed(toColorImage(fg), markers)
        #self.toc('ws')
        markers[markers==1] = 0
        markers[markers>1] = 255
        markers[markers==-1] = 0
        markers = toByteImage(markers)
        markers = cv2.erode(markers, 
                            cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
        markers = cv2.GaussianBlur(toGrayImage(markers), (3,3), 2)
        return markers

    def __analyse(self, detected_circles, value: B1CImage) -> GoBoardClassification:
        val = np.ones(self.size**2)*2
        num_val = np.zeros(self.size**2)
        detections = []
        for circles in detected_circles:
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles:
                    on_grid = (self.BOARD.go_board_shifted[:,0] - i[0])**2 + (self.BOARD.go_board_shifted[:,1] - i[1])**2 < (i[2]/2)**2
                    mask = np.zeros_like(value)
                    if np.any(on_grid):
                        idx = np.argwhere(on_grid)
                        for idx_ in idx:
                            coord = self.BOARD.go_board_shifted[idx_]
                            detections.append(np.squeeze(coord).astype(int))

                        cv2.circle(mask, (i[0], i[1]), int(i[2]/2), (255), -1)

                        patch = value[mask.astype(bool)]
                        if np.median(patch) < 127:
                            val[np.argwhere(on_grid)] = 1
                        elif np.median(patch > 127):
                            val[np.argwhere(on_grid)] = 0
                        num_val[np.argwhere(on_grid)] = np.median(patch)

                        if self.debugStatus(debugkeys.Detected_Intensities):
                            cv2.circle(self.img_debug, (i[0], i[1]), int(i[2]/2), (0, 255, 0), 1)
                            cv2.putText(self.img_debug, "{}".format(np.median(patch)),
                                    (int(i[0]), int(i[1])),
                                    self.cv_settings.font, 
                                    self.cv_settings.fontScale,
                                    self.cv_settings.fontColor,
                                    self.cv_settings.thickness,
                                    self.cv_settings.lineType)
                            self.showDebug(debugkeys.Detected_Intensities, self.img_debug)
        detections = np.array(detections)
        return val, detections

