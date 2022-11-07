from distutils import dep_util
from distutils.log import debug
from xml.etree.ElementInclude import include
import numpy as np
import cv2
from inpoly import inpoly2
from lu_vp_detect import VPDetection, LS_ALG
from skimage import data, color, img_as_ubyte, exposure, transform, img_as_float
from pygo.utils.debug import DebugInfo, DebugInfoProvider, Timing
from pygo.utils.homography import compute_homography_and_warp, compute_homography
import matplotlib.pyplot as plt
import math
from pygo.utils.line import intersect, is_line_within_board, warp_lines
from pygo.utils.misc import *
from pygo.utils.image import toByteImage, toCMYKImage, toGrayImage, toYUVImage
from pygo.utils.plot import Plot
from pygo.Signals import *
from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration
from functools import partial
from shapely.geometry import Polygon, LineString, Point
from pygo.CameraCalib import CameraCalib
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import pdb
import logging
from pygo.utils.typing import B1CImage, B3CImage, Point2D, Point3D, Image, Mask, B1CImage, Corners
from pygo.utils.image import toColorImage
from nptyping import NDArray
from enum import Enum, auto
from pudb import set_trace

class debugkeys(Enum):
    Detected_Lines = auto()
    Detected_Grid = auto()
    Affine_Registration = auto()
    Board_Outline = auto()

class NoVanishingPointsDetectedException(Exception):
    pass

class GoBoard(DebugInfoProvider, Timing):
    def __init__(self, camera_calibration: CameraCalib):
        DebugInfoProvider.__init__(self)
        Timing.__init__(self)
        self.camera_calibration = camera_calibration
        self.H = np.eye(3)
        self.vp = VPDetection(focal_length=self.camera_calibration.get_focal(), 
                              principal_point=self.camera_calibration.get_center(), 
                              length_thresh=50,
                              line_search_alg=LS_ALG.LSD)
        self.current_unwarped = None
        self.grid = None
        self.go_board_shifted = None
        self.hasEstimate = False
        self.grid_lines = None
        #self.border_size = 15
        self.border_size = 30
        self.plot = Plot()

        for key in debugkeys:
            self.available_debug_info[key.name] = False

        Signals.subscribe(OnCameraGeometryChanged, self.cameraGeometryHasChanged)
        Signals.subscribe(OnInputChanged, self.reset)

    def __calib(self, *args) -> None:
        self.calib()

    def cameraGeometryHasChanged(self, *args) -> None:
        self.vp = VPDetection(focal_length=self.camera_calibration.get_focal(), 
                              principal_point=self.camera_calibration.get_center(), 
                              length_thresh=50,
                              line_search_alg=LS_ALG.LSD)

    def reset(self, *args) -> None:
        self.vp = VPDetection(focal_length=self.camera_calibration.get_focal(), 
                              principal_point=self.camera_calibration.get_center(), 
                              length_thresh=50,
                              line_search_alg=LS_ALG.LSD)

        self.grid = None
        self.go_board_shifted = None
        self.hasEstimate = False
        self.grid_lines = None
        self.H = np.eye(3)

    def crop(self, pts : Corners, img : Image) -> Mask:
        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        croped = img[y:y+h, x:x+w].copy()

        ## (2) make mask
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        ## (4) add the white background
        bg = np.ones_like(croped, np.float32)
        bg[:] = np.nan
        cv2.bitwise_not(bg,bg, mask=mask)
        dst2 = bg+ dst
        return dst2

    def imgToPatches(self, img : Image) -> List[Image]:
        patches = []
        for path in zip(self.cl2, self.ct2, self.cr2, self.cb2):
            #l,r,t,b):
            l1 = np.array([path[0][0], path[1][1]])
            l2 = np.array([path[2][0], path[1][1]])
            l3 = np.array([path[2][0], path[3][1]])
            l4 = np.array([path[0][0], path[3][1]])
            p = np.array([l1,l2,l3,l4]).astype(int)

            patch = self.crop(p, img)

            # fail for false calib -> None
            if not np.all(np.array(patch.shape) > 0):
                return None

            patch =  transform.resize(patch, (32,32),  anti_aliasing=True)
            patches.append(patch)
        return patches

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
        img_gray = toYUVImage(img)[:,:,0]
        img_bw = cv2.adaptiveThreshold(img_gray,
                                        255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                        cv2.THRESH_BINARY_INV,
                                        5,
                                        15)

        try:
            vp1, vp2 = self.get_vp(img_gray)
        except NoVanishingPointsDetectedException:
            return img

        # assumption most lines in the image are from the go board -> vp give us the plane
        # the contour which belongs to those vp is the board
        corners = self.detect_board_corners(vp1, vp2, img_bw)
        img_ = img.copy()
        if corners is not None:
            corners = np.int32([corners])
            cv2.polylines(img_, corners, color=(0,255,0), isClosed=True, thickness=3)
        return img_


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
    
    def calib(self, img: Image) -> None:
        #img_c = img
        if len(img.shape) == 3:
            h,w,c = img.shape
            #instead of the grayscale variant extract the red part
            img_bw = toYUVImage(img)[:,:,0]
            img = toCMYKImage(img)[:,:,3]
            #plt.imshow(img)
            #plt.show()
        else:
            h,w = img.shape

        try:
            vp1, vp2 = self.get_vp(img) 
        except NoVanishingPointsDetectedException:
            return False

        #thresh, img_bw = cv2.threshold(img, \
        #                            0, \
        #                            255, \
        #                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_bw = cv2.adaptiveThreshold(img_bw, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5,15)

        #cv2.imshow('bw',img_bw)
        #cv2.waitKey(1)
        # clean img_bw


        # assumption most lines in the image are from the go board -> vp give us the plane
        # the contour which belongs to those vp is the board
        corners = self.detect_board_corners(vp1, vp2, img_bw.copy())    
        if corners is None:
            logging.error("Could not detect Go-Board corners!")
            return

        #self.grid = get_ref_go_board_coords(np.min(corners, axis=0), 
        #                                    np.max(corners, axis=0),
        #                                    self.border_size)
        self.grid = get_ref_coords(img.shape, self.border_size)

        target_corners = np.vstack((self.grid[18], 
                                    self.grid[0], 
                                    self.grid[342],
                                    self.grid[360]))

        
        H, ret = cv2.findHomography(target_corners, corners)
        self.img_limits = (img.shape[1],img.shape[0])
        
        self.H = H 
        self.grid_lines, self.grid_img, self.grd_overlay = get_grid_lines(self.grid)
        self.hasEstimate=True
        img_c_trim, (x,y) = mask_board(img, self.grid, self.border_size)
        self.go_board_shifted = self.grid - np.array([x,y])
        self.cell_w = np.mean(np.diff(self.go_board_shifted.reshape(19,19,2)[:,:,0], axis=0))
        self.cell_h = np.mean(np.diff(self.go_board_shifted.reshape(19,19,2)[:,:,1], axis=1))
        logging.info('Grid width {}'.format(self.cell_w))
        logging.info('Grid height {}'.format(self.cell_h))

        self.cl2 = self.go_board_shifted - np.array([self.cell_w/2, 0])
        self.cr2 = self.go_board_shifted + np.array([self.cell_w/2, 0])
        self.ct2 = self.go_board_shifted + np.array([0, self.cell_h/2])
        self.cb2 = self.go_board_shifted - np.array([0, self.cell_h/2])


        logging.debug(self.H)
        logging.info("Calibration successful")
        Signals.emit(OnGridSizeUpdated, self.cell_w, self.cell_h)
        Signals.emit(OnBoardDetected, self.extract(img) , corners, self.H)
        Signals.emit(OnBoardGridSizeKnown, self.go_board_shifted)



    def check_patches_are_centered(self, img: Image) -> bool:
        cropped = self.extract(img)
        patches = self.imgToPatches(cropped)
        if patches is None:
            logging.warning('could not extract patches')
            return False

        lines_x = []
        lines_y = []
        for i, patch in enumerate(patches):
            x,y = np.unravel_index(i, (19,19))
            if  x in [0,18] or y in [0,18]:
                #skip corners due to possible inclusion of board corners
                continue
            if x in [2,4,8,10,14,16] and y in [3,9,15]:
                #exclude calib patches
                continue
            thresh, patch_bw = cv2.threshold(toByteImage(patch), \
                                    0, \
                                    255, \
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            lines_x.append(np.argmax(np.sum(patch_bw, 0)))
            lines_y.append(np.argmax(np.sum(patch_bw, 1)))
        
        if np.std(lines_x) > 2.8 or np.std(lines_y) > 2.8:
            logging.debug("std x: {}".format(np.std(lines_x)))
            logging.debug("std y: {}".format(np.std(lines_y)))
            return False
        else:
            return True
 
    def extract(self, img):
        self.current_unwarped = img
        img_w = cv2.warpPerspective(img, np.linalg.inv(self.H), self.img_limits)

        #cv2.imshow('PyGO', img_w) 
        #cv2.waitKey(1)

        img_c_trim, (x,y) = mask_board(img_w, self.grid, self.border_size)
        #self.go_board_shifted = self.grid - np.array([x,y])
        #wide_limits = (self.img_limits[0]+50, self.img_limits[1]+50)
        #wide_h = np.linalg.inv(self.H)
        #wide_h[0,2] += 25
        #wide_h[1,2] += 25
        #img_w = cv2.warpPerspective(img, wide_h, wide_limits)
        #img_c_wide, _ = mask_board(img_w, self.grid, wide_limits[0])
        #cv2.imshow('PyGO', img_c_trim) 
        #cv2.waitKey(100)


        return img_c_trim#, img_c_wide

    def extractOnPoints(self, img):
        img = self.extract(img)
        '''
            + = empty
            B = Black
            W = White
            O = Ref points on board

            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + B O W + + + B O W + + + B O W + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + B O W + + + B O W + + + B O W + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + B O W + + + B O W + + + B O W + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
        '''
        points_w = np.array([[ 4, 3],
                             [ 4, 6],
                             [ 4, 9],
                             [ 4,12],
                             [ 4,15],
                             [10, 3],
                             [10, 6],
                             [10, 9],
                             [10, 12],
                             [10,15],
                             [16, 3],
                             [16, 6],
                             [16, 9],
                             [16,12],
                             [16,15]])
        points_b = np.array([[ 2 ,3],
                             [ 2 ,6],
                             [ 2 ,9],
                             [ 2,12],
                             [ 2,15],
                             [ 8 ,3],
                             [ 8 ,6],
                             [ 8 ,9],
                             [ 8,12],
                             [ 8,15],
                             [14 ,3],
                             [14 ,6],
                             [14 ,9],
                             [14,12],
                             [14,15]])

        corners = [np.array([0,0]),
                   np.array([0,18]),
                   np.array([18,0]),
                   np.array([18,18])]
        idx_ = np.arange(1,18)
        edges_t = [np.array([0,i]) for i in idx_]
        edges_l = [np.array([i,0]) for i in idx_]
        edges_r = [np.array([18,i]) for i in idx_]
        edges_b = [np.array([i,18]) for i in idx_]
        edges = edges_b + edges_l + edges_r + edges_t

        idx_corner = [np.ravel_multi_index(arr, (19,19)) for arr in corners]
        idx_edge   = [np.ravel_multi_index(arr, (19,19)) for arr in edges]


        idx_w = np.ravel_multi_index(points_w.T, (19,19))
        idx_b = np.ravel_multi_index(points_b.T, (19,19))
        idx_n = np.arange(19*19)
        idx_n = np.delete(idx_n, np.concatenate((idx_w, idx_b, idx_corner, idx_edge)))
        patches = [[],[],[],[],[]]
        p = self.imgToPatches(img)
        for c, idx in enumerate([idx_w, idx_b, idx_n, idx_edge, idx_corner]):
            patches[c].append(np.array(p)[idx])
            #plt.imshow(np.vstack(np.array(p)[idx])/255)
            #plt.show()
        return patches








''''

    def calib_old(self, img: Image) -> None:
        #img_c = img
        if len(img.shape) == 3:
            h,w,c = img.shape
            #instead of the grayscale variant extract the red part
            img = toCMYKImage(img)[:,:,3]
            #plt.imshow(img)
            #plt.show()
        else:
            h,w = img.shape

        try:
            vp1, vp2 = self.get_vp(img) 
        except NoVanishingPointsDetectedException:
            return False

        thresh, img_bw = cv2.threshold(img, \
                                    0, \
                                    255, \
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow('bw',img_bw)
        #cv2.waitKey(1)
        # clean img_bw


        # assumption most lines in the image are from the go board -> vp give us the plane
        # the contour which belongs to those vp is the board
        corners = self.detect_board_corners(vp1, vp2, img_bw.copy())
        if corners is None:
            logging.warning('Calib Failed - No corners detected')
            return
        mask = np.zeros((img_bw.shape),np.uint8)
        mask = cv2.fillConvexPoly(mask, corners.astype(int), 255)

        # rectify image
        lines_vp = []
        vps_list_1 = []
        vps_list_2 = []

        vps_list_1.append(vp1)
        vps_list_2.append(vp2)        
        lines_cluster = self.vp.get_lines()
        # drop lines with least 
        del lines_cluster[(np.argmin([len(x) for x in lines_cluster]))]
        lines_vp.append(lines_cluster)

        vp1 = np.median(np.array(vps_list_1), axis=0)
        vp2 = np.median(np.array(vps_list_2), axis=0)
        img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_c = cv2.polylines(img_c, np.array([corners.astype(int)]), True, (0, 255, 255), 1)

        H, img_limits = compute_homography(img_c, vp1, vp2, clip=False, clip_factor=1)
        lines_v = self.vp.lines_v
        lines_h = self.vp.lines_h


        for l in lines_v:
            pt1 = l[:2].astype(int)
            pt2 = l[2:].astype(int)
        for l in lines_h:
            pt1 = l[:2].astype(int)
            pt2 = l[2:].astype(int)
 
        self.img_limits = img_limits

        # remove all lines which are outside of our detected board

        img_lines = cv2.warpPerspective(img_c.copy(), H, img_limits)
        corners_w = warp_lines(corners, H)
        poly = Polygon(corners_w)

        lines_v_w = warp_lines(lines_v.reshape(-1,2), H).reshape(-1,4)
        lines_h_w = warp_lines(lines_h.reshape(-1,2), H).reshape(-1,4)

        lines = intersect(np.array(lines_v_w), np.array(lines_h_w))

        lines_v__ = []
        lines_h__ = []
        for line in lines_v_w:
            pt1 = line[:2]
            pt2 = line[2:]
            vec = pt1-pt2
            if poly.crosses(LineString([pt1+vec,pt2-vec])):
                lines_v__.append(line)
                if self.debugStatus(debugkeys.Detected_Lines):
                    cv2.line(img_lines, pt1.astype(int), pt2.astype(int), (255,0,0), 2)
        for line in lines_h_w:
            pt1 = line[:2]
            pt2 = line[2:]
            vec = pt1-pt2
            if poly.crosses(LineString([pt1+vec,pt2-vec])):
                lines_h__.append(line)
                if self.debugStatus(debugkeys.Detected_Lines):
                    cv2.line(img_lines, pt1.astype(int), pt2.astype(int), (0,255,0), 2)


        lines_v__ = np.array(lines_v__)
        lines_h__ = np.array(lines_h__)
        lines_v__ = lines_v_w
        lines_h__ = lines_h_w

        lines = intersect(lines_v__, lines_h__)
        lns = np.concatenate((lines_v__, lines_h__), axis=0)
        cns = np.concatenate((corners_w, np.roll(corners_w, 1, 0)), axis=1)
        lines = np.concatenate((lines, intersect(lns, cns)))
        lines_cleaned =[]
        for pt in lines:
            if Point(pt[0],pt[1]).within(poly):
                lines_cleaned.append(pt)
        for pt in lines_cleaned:
            cv2.circle(img_lines, pt.astype(int), 1, (0,0,255))

        self.showDebug(debugkeys.Detected_Lines, img_lines)


        lines_cleaned = np.array(lines_cleaned)

        self.grid = get_ref_go_board_coords(np.min(lines_cleaned, axis=0), 
                                            np.max(lines_cleaned, axis=0))
       #[plt.axline((l[0],l[1]), (l[2],l[3])) for l in lines_v__]
        #[plt.axline((l[0],l[1]), (l[2],l[3])) for l in lines_h__]
        #plt.scatter(lines[:,0], lines[:,1])
        #plt.show()

 

        # init ref grid
        if len(lines_cleaned) == 0:
            logging.warning('Could not find enought lines - calib failed')
            return 
        
        if len(lines_cleaned) > 5000:
            logging.warning('Too Many lines detected! - calib failed')
            return 


        # warp back to original images
        lines_raw_orig = cv2.perspectiveTransform(lines_cleaned[:,None,:], 
                                                    np.linalg.inv(H)).squeeze()


    def visualize(iteration, error, X, Y, ax):
            #if iteration % 50 == 0:
                plt.cla()
                ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
                ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
                plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
                    iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
                ax.legend(loc='upper left', fontsize='x-large')
                plt.draw()
                plt.pause(0.001)

        
        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        callback = partial(visualize, ax=fig.axes[0])
        reg = AffineRegistration(**{'X': lines_cleaned, 
                                'Y': self.grid, 
                                'max_iterations': 1200,
                                'tolerance': 0.0001,
                                })

        if self.debugStatus(debugkeys.Affine_Registration):
            ty, param = reg.register(callback)
        else:
            ty, param = reg.register()

        R = np.eye(3)
        R[0:2,0:2]=param[0]
        R[0:2,2]=param[1]

        mask = cv2.warpPerspective(mask, H, self.img_limits)
        mask = cv2.dilate(mask, np.ones((3,3)), iterations=15)
        rect = cv2.boundingRect(mask.astype(np.uint8))
        offset_x, offset_y,offset_w,offset_h = rect
        img_c_w = cv2.warpPerspective(img_c, H, self.img_limits)
        img_c_w = cv2.cvtColor(img_c_w, cv2.COLOR_BGR2GRAY)
        #plt.imshow(np.dstack((mask, img_c_w, mask)))
        #plt.show()

        w_board = cv2.transform(np.array([self.grid]), (R))[0][:,:2]
        ow_board = cv2.perspectiveTransform(np.array([self.grid]), R@np.linalg.inv(H))[0]

        src_pt = find_src_pt(ow_board, lines_raw_orig)
        H_refined = cv2.findHomography(src_pt, ow_board)[0]
        w_board = cv2.perspectiveTransform(np.array([self.grid]), R @ np.linalg.inv(H) @ np.linalg.inv(H_refined))[0]

        H_refined = R @ np.linalg.inv(H) @ np.linalg.inv(H_refined)
        
        if self.debugStatus(debugkeys.Detected_Grid):
            img_grid = self.plot.plot_grid(img.copy(), w_board.reshape(-1,2))
            self.showDebug(debugkeys.Detected_Grid, img_grid)

        self.H = H_refined 
        if self.check_patches_are_centered(img):
            #determined spaces from grid spacing
            self.grid_lines, self.grid_img, self.grd_overlay = get_grid_lines(self.grid)
            self.hasEstimate=True
            logging.debug(self.H)
            logging.info("Calibration successful")
            OnBoardDetected.emit(self.extract(img) , corners, self.H)
        else:
            self.H = np.eye(3)
            logging.info('Calibration failed! - ')


'''