from argparse import ArgumentError
import cv2
import logging
import numpy as np
import sklearn
from enum import Enum, auto

from pycpd import AffineRegistration
try:
    from lu_vp_detect import VPDetection, LS_ALG
    VP_MODULE = True
except ImportError:
    VP_MODULE = False
from skimage import transform
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from nptyping import NDArray
from typing import Optional, Tuple, List

from pygo.utils.debug import DebugInfoProvider, Timing, debugkeys
from pygo.utils.misc import *
from pygo.utils.image import toByteImage, toCMYKImage, toGrayImage, toYUVImage
from pygo.utils.plot import Plot
from pygo.Signals import *
from pygo.CameraCalib import CameraCalib
from pygo.utils.typing import B1CImage, B3CImage, Point2D, Point3D, Image, Mask, B1CImage, Corners
from pygo.utils.image import toColorImage
from pygo.Settings import CORNER_DETECTION_ALG, PyGOSettings

class NoVanishingPointsDetectedException(Exception):
    pass

class GoBoard(DebugInfoProvider, Timing):
    def __init__(self, 
                 camera_calibration: Optional[CameraCalib]):
        DebugInfoProvider.__init__(self)
        Timing.__init__(self)
        self.camera_calibration = camera_calibration
        self.vp = None
        if self.camera_calibration is not None and VP_MODULE:
            self.vp = VPDetection(focal_length=self.camera_calibration.get_focal(), 
                              principal_point=self.camera_calibration.get_center(), 
                              length_thresh=50,
                              line_search_alg=LS_ALG.LSD)
        if PyGOSettings['CornerDetectionAlg'] == CORNER_DETECTION_ALG.WITH_VP and self.vp is None:
            raise RuntimeWarning("You selected a corner algorithm which requieres a camera configuration. Cameras configuration was not provided")
 
        self.H = np.eye(3)
        self.hasEstimate = False

        # Coordinate grids for different view
        self.grid = None
        self.go_board_shifted = None
        self.grid_lines = None
        self.cell_w = None
        self.cell_h = None
        self.img_limits = None

        # Images
        self.current_unwarped = None
        self.grd_overlay = None
        self.grid_img = None

        # Settings
        self.border_size = 30
        self.binarization_kernel_size = 51
        self.binarization_with_morphological_closing = False

        for key in debugkeys:
            self.available_debug_info[key.name] = False

        CoreSignals.subscribe(OnCameraGeometryChanged, self.camera_geometry_has_changed)
        CoreSignals.subscribe(OnInputChanged, self.reset)
        CoreSignals.subscribe(DetectBoard, self.__calib)
        CoreSignals.subscribe(OnSettingsChanged, self.__update_settings)

    def __update_settings(self, args) -> None:
        pass

    def __calib(self, args) -> None:
        img =args[0]
        self.calib(img)

    def camera_geometry_has_changed(self, *args) -> None:
        if not VP_MODULE:
            logging.warning("Vanishing Point module not available -> vanishing point detection disabled")
        elif self.camera_calibration is None:
            logging.warning("No camera calibration provided -> vanishing point detection disabled")
        else:
            self.vp = VPDetection(focal_length=self.camera_calibration.get_focal(), 
                              principal_point=self.camera_calibration.get_center(), 
                              length_thresh=50,
                              line_search_alg=LS_ALG.LSD_WITH_MERGE)


    def reset(self, *args) -> None:
        if not VP_MODULE:
            logging.warning("Vanishing Point module not available -> vanishing point detection disabled")
        elif self.camera_calibration is None:
            logging.warning("No camera calibration provided -> vanishing point detection disabled")
        else:
            self.vp = VPDetection(focal_length=self.camera_calibration.get_focal(), 
                              principal_point=self.camera_calibration.get_center(), 
                              length_thresh=50,
                              line_search_alg=LS_ALG.LSD_WITH_MERGE)

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


    def line_point_distance(self, point: Point2D, line_s: Point2D, line_e: Point2D) -> float:
        '''
            calculates the perpendicular distance of a point towards a line given by its
            start and endpoints
        '''
        nom = np.abs((line_e[0]-line_s[0])*(line_s[1]-point[1]) - 
                      (line_s[0]-point[0])*(line_e[1]-line_s[1]))

        denom = np.sqrt((line_e[0]-line_s[0])**2 + 
                        (line_e[1]-line_s[1])**2) + 1e-12
        return nom/denom


    def sort_corners(self, corners: Corners) -> Optional[NDArray]:
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


    def get_corners(self, img:Image, vp1: Optional[Point2D], vp2: Optional[Point2D]) -> NDArray:
        '''
        vp1: 2d vanishing point
        vp2: 2d vanishing point
        img: thresholded image of the board
        vp1 and vp2 can be omitted for speadup
        pro: super fast
        con: fails when we have stones in the corner
        '''
        contours, _ = cv2.findContours(img, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)

        #TODO: determine min_dist based on image geometry
        min_dist = 10000
        corners = []
        corners_mat = None

        # we expect the go board to cover at least a quater of the smaller image
        # frames side
        smaller_side = min(img.shape)
        min_area = (0.25*smaller_side)**2

        for c, cnt in enumerate(contours):
            if len(cnt) < 4:
                # contours with less than four points can be skipped
                continue

            area = cv2.contourArea(cnt)
            if area > min_area:
                # somethimes the contour has small dents .. approximate till
                # we have only four corners left
                for eps in np.linspace(0.001, 0.05, 10):
                    # approximate the contour
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)

                    if len(approx) == 4:
                        # we found an approximation with four corneres we can stop
                        cnt = approx
                        break

                if len(approx) != 4:
                    # in case we looped to the end without a good result goto 
                    # next shape
                    logging.debug("False corner count, we have : {}".format(len(approx)))
                    continue

                # Find corners on the mask using the four most prominent corners
                # should do the trick
                mask = np.zeros((img.shape),np.uint8)
                mask = cv2.fillConvexPoly(mask, np.array(approx), 255)

                corners = cv2.goodFeaturesToTrack(mask, 
                                                    maxCorners=4, 
                                                    qualityLevel=0.1, 
                                                    minDistance=200)
                if corners is None:
                    # try the next shape
                    continue
                
                # sort corners clockwise
                corners = self.sort_corners(corners)
                
                if corners is not None and len(corners) >= 4:
                    (topmost, rightmost, bottommost, leftmost) = corners

                    # check the difference between the approximated and original mask
                    # when we have masked the go board a four corner approximation 
                    # should be very close to the actual mask
                    corner_mask = np.zeros((img.shape), np.uint8)
                    corner_mask = cv2.fillConvexPoly(corner_mask, 
                                                        corners[:,None,:].astype(int), 
                                                        255)
                    dev_pixels = np.sum(mask-corner_mask)

                    if dev_pixels > 0.2 * np.sum(mask):
                        # the approximated mask deviates to much from the original
                        # mask we have mask to much area, skip to the next shape
                        continue
                    
                    # given two vanishing points in our image we can check wether
                    # the corners match up with the vanishing points
                    # only works when most lines in the image are from the go board
                    # which should be a fair assumption
                    if vp1 is not None and vp2 is not None:
                        d1 = min(self.line_point_distance(vp1, leftmost, topmost), 
                                self.line_point_distance(vp2, leftmost, topmost))
                        d2 = min(self.line_point_distance(vp1, rightmost, bottommost),
                                self.line_point_distance(vp2, rightmost, bottommost))

                        d3 = min(self.line_point_distance(vp2, rightmost, topmost),
                                self.line_point_distance(vp1, rightmost, topmost))
                        d4 = min(self.line_point_distance(vp2, leftmost, bottommost),
                                self.line_point_distance(vp1, leftmost, bottommost))

                        # the best detection has minimal deviation from vp 
                        # and the least deviation between the four corners area 
                        # and the poly area
                        dist = d1 + d2 + d3 + d4
                        if dist < min_dist:
                            min_dist = dist
                            corners_mat = np.array([leftmost, topmost, rightmost, bottommost])
                    else:
                        # fast version without vp check
                        corners_mat = np.array([leftmost, topmost, rightmost, bottommost])

        return corners_mat


    def detect_board_corners_fast(self, img: B3CImage, vp1: Point2D=None, vp2: Point2D=None) -> NDArray:
        if vp1 is None:
            logging.debug('Running corner detection WITHOUT vanishing point module')

        img_bw = self.binarizeImage(img, C=10)
        corners = self.get_corners(img_bw, vp1, vp2)
        corners = self.refine_corners(corners, img)
        if corners is not None:
            self.update_grid(img_bw, corners)
            if self.check_board_alignment(img_bw):
                # stop search when the have found a good solution
                logging.info("Board position found")
                self.hasEstimate=True


        logging.debug('Corners {}'.format(corners))
 
        return corners


    def update_grid(self, img: Image, corners: NDArray) -> None:
        '''
            Updates all relevant data used for transformation of image
            and the grid coordinates used during further processing
        '''
        print('Updating grid')
        self.grid = get_ref_coords(img.shape, self.border_size)
        self.img_limits = (img.shape[1],img.shape[0])

        target_corners = np.vstack((self.grid[18], 
                                    self.grid[0], 
                                    self.grid[342],
                                    self.grid[360]))

        self.H, _ = cv2.findHomography(target_corners, corners)

        _, (x,y) = mask_board(img, self.grid, self.border_size)
        self.grid_lines, self.grid_img, self.grd_overlay = get_grid_lines(self.grid)

        self.go_board_shifted = self.grid - np.array([x,y])

        self.cell_w = np.mean(np.diff(self.go_board_shifted.reshape(19,19,2)[:,:,0], axis=0))
        self.cell_h = np.mean(np.diff(self.go_board_shifted.reshape(19,19,2)[:,:,1], axis=1))


    def detect_board_corners(self, vp1: Point2D, vp2: Point2D, img: B3CImage) -> Optional[NDArray]:
        '''
            Returns the corner coordintes of the detected go board as 2d array or None 
            when no detection could be made
        '''
        for C in np.arange(1,50,5):
            # test different thresholds as different illumination conditions demand
            # different settings
            img_bw = self.binarizeImage(img, C)
            corners = self.get_corners(img_bw, vp1, vp2)
            corners = self.refine_corners(corners, img)

            if corners is not None:
                self.update_grid(img_bw, corners)
                if self.check_board_alignment(img_bw):
                    # stop search when the have found a good solution
                    logging.info("Board position found")
                    self.hasEstimate=True
                    break

        #if corners is None:
        #    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        #    # loop over the kernels sizes
        #    for  in range(0):
        #        cv2.morphologyEx(src=img_bw, dst=img_bw, op=cv2.MORPH_CLOSE, kernel=kernel)
        #        corners = self.get_corners(vp1, vp2, img_bw)
        #        #up to threee filter layers
        #        if corners is not None:
        #           break

        logging.debug('Corners {}'.format(corners))
 
        return corners


    def binarizeImage(self, img:B3CImage, C:int=20) -> B1CImage:
        '''
            Returns a binarized image which should clearly show the boards grid
        '''
        img_gray = toYUVImage(img)[:,:,0]
        #clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        #img_gray = clahe.apply(img_gray)
        #img_gray = cv2.equalizeHist(img_gray)
        img_bw = cv2.adaptiveThreshold(img_gray,
                                        255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                        cv2.THRESH_BINARY_INV,
                                        self.binarization_kernel_size,
                                        C)

        if self.binarization_with_morphological_closing:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            cv2.morphologyEx(src=img_bw, dst=img_bw, op=cv2.MORPH_CLOSE, kernel=kernel)

        return img_bw


    def get_corners_overlay(self, img: B3CImage) -> NDArray:
        '''
            Plotting the detection frame onto the image uses a reduced setting
            compared to the actual calibration for speedup. Thus detecion is possible
            event if the green border is not shown.
        '''

        # for fast binarization we use a preset threshold, this can fail on extreme 
        # illuminations
        corners = self.track_corners(img)
        img_ = img.copy()

        if corners is not None:
            corners = np.int32([corners])
            cv2.polylines(img_, corners, color=(0,255,0), isClosed=True, thickness=3)

            if self.debugStatus(debugkeys.Board_Outline):
                img_bw = self.binarizeImage(img, C=20)
                img_bw_ = toColorImage(img_bw)
                cv2.polylines(img_bw_, corners, color=(0,255,0), isClosed=True, thickness=3)
                self.showDebug(debugkeys.Board_Outline, img_bw_)
 
        return img_


    def track_corners(self, img: B3CImage) -> NDArray:
        # for fast binarization we use a preset threshold, this can fail on extreme 
        # illuminations
        #if self.corner_detection_alg == CORNER_DETECTION_ALG.WITH_VP 
        corners = self.detect_board_corners_fast(img=img)
        return corners


    def get_vp(self, img: B1CImage) -> Tuple[Point3D, Point3D]:
        '''
            returns the vertical and horizontal vanishing points (3D)
            from a given image
        '''
        van_points = self.vp.find_vps(img)
        if van_points is None:
            raise NoVanishingPointsDetectedException()

        vps = self.vp.vps_2D
        vp3d = self.vp.vps
        vert_vp = np.argmax(np.abs(vp3d[:,2]))
        vps = np.delete(vps, vert_vp, axis=0)
        vp1 = np.array([vps[0,0], vps[0,1], 1])
        vp2 = np.array([vps[1,0], vps[1,1], 1])
        return vp1, vp2
    

    def calib(self, img: B3CImage) -> None:
        '''
            Detect the board and signal other components (UI)
        '''
        corners = None
        img_c = img
        img = toCMYKImage(img)[:,:,3]
        logging.info("Detecting Go-Board...")
        logging.info('Using corner detection algorithm: {}'.format(PyGOSettings['CornerDetectionAlg']))
        if PyGOSettings['CornerDetectionAlg'] == CORNER_DETECTION_ALG.WITH_VP:
            try:
                # assumption most lines in the image are from the go board 
                # -> vp give us the plane
                # the contour which belongs to those vp is the board
                vp1, vp2 = self.get_vp(img) 
                corners = self.detect_board_corners(vp1=vp1, vp2=vp2, img=img_c)
            except NoVanishingPointsDetectedException:
                return False

        elif PyGOSettings['CornerDetectionAlg'] == CORNER_DETECTION_ALG.FAST:
            corners = self.detect_board_corners_fast(img_c)
        elif PyGOSettings['CornerDetectionAlg'] == CORNER_DETECTION_ALG.CPD:
            corners = self.detect_board_cpd(img_c)
            
        if corners is None:
            logging.error("Could not detect Go-Board corners!")
            return

        logging.debug('Grid width {}'.format(self.cell_w))
        logging.debug('Grid height {}'.format(self.cell_h))

        logging.debug(self.H)
        logging.info("Board detected")

        CoreSignals.emit(OnGridSizeUpdated, self.cell_w, self.cell_h)
        CoreSignals.emit(OnBoardDetected, self.extract(img) , corners, self.H)
        UISignals.emit(UIOnBoardDetected, self.extract(img) , corners, self.H)
        CoreSignals.emit(OnBoardGridSizeKnown, self.go_board_shifted)

    def detect_board_cpd(self, img: B3CImage) -> NDArray:
        lsd = cv2.createLineSegmentDetector()
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lines = lsd.detect(img_bw)[0].squeeze()
        points = []
        for line in lines:
            points.append(np.array([line[0], line[1]]))
            points.append(np.array([line[2], line[2]]))

        points = np.asarray(points) 
        
        orientation = []
        #find two major orientations
        for line in lines:
            orientation.append(np.arctan2(abs(line[3]-line[1]), abs(line[2]-line[0]))*180/np.pi)

        lines_v = []
        for o,l in zip(orientation, lines):
            if o < 5:
                lines_v.append(l)

        orientation = []
        #find two major orientations
        for line in lines:
            orientation.append(np.arctan2(abs(line[3]-line[1]), abs(line[2]-line[0]))*180/np.pi)

        lines_h = []
        for o,l in zip(orientation, lines):
            if o >85 and o<95:
                lines_h.append(l)
                
        new_lines_v = [self.make_linesegment_longer(line) for line in lines_v]
        new_lines_h = [self.make_linesegment_longer(line) for line in lines_h]
        intersections = []
        for lv in new_lines_v:
            for lh in new_lines_h:
                intersections.append(self.get_line_intersection(lv, lh))

        intersections = [x for x in intersections if x is not None]
        km = sklearn.cluster.KMeans(n_clusters=min(19*19, len(intersections)))
        km.fit(intersections)
        intersections = km.cluster_centers_
        w,h = img_bw.shape[1], img_bw.shape[0]
        x_min = min([pt[0] for pt in intersections])
        x_max = max([pt[0] for pt in intersections])
        y_min = min([pt[1] for pt in intersections])
        y_max = max([pt[1] for pt in intersections])
        len_x = x_max - x_min
        len_y = y_max - y_min

        board_ref = []
        for i in range(19):
            for j in range(19):
                board_ref.append((x_min + i*(len_x/19), y_min + j*(len_y/19)))
        board_ref = np.asarray(board_ref)
        src = np.asarray(intersections)
        noise = 1-(19*19)/len(src)
        reg = AffineRegistration(X=src, Y=board_ref)
        pt, params = reg.register()
        
        corners = pt[np.array([18,0, 18*19, 18*19+18])]
        print(corners)
        if corners is not None:
            self.update_grid(img_bw, corners)
            if self.check_board_alignment(img_bw):
                # stop search when the have found a good solution
                logging.info("Board position found")
                self.hasEstimate=True

        return corners
   

    def make_linesegment_longer(self, line):
        orientation = np.arctan2(line[3] - line[1], line[2]-line[0])
        length = np.sqrt((line[3]-line[1])**2 + (line[2]-line[0])**2)
        f = 0.08
        c = np.cos(orientation) * f * length
        s = np.sin(orientation) * f * length
        return [line[0] - c , line[1] - s, line[2] + c, line[3] + s]
    
    def get_line_intersection(self, line_a, line_b):
        p0_x, p0_y, p1_x, p1_y = line_a
        p2_x, p2_y, p3_x, p3_y = line_b

        s1_x = p1_x - p0_x;     s1_y = p1_y - p0_y
        s2_x = p3_x - p2_x;     s2_y = p3_y - p2_y

        s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y)
        t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y)

        if (s >= 0 and s <= 1 and t >= 0 and t <= 1):
            return p0_x + (t * s1_x), p0_y + (t * s1_y)

        return None
            

    def check_board_alignment(self, img:Image) -> bool:
        '''
            Perfectl aligned [extracted] images of the go board should have vertical 
            and horizontal lines in the cropped image. We test this by looking at a 
            binarized version of the board which should have distinct maxima when 
            summed vertically/horizontally
        '''

        cropped = self.extract_borderless(img)
        # crop bordering lines
        cw = int(self.cell_w //2)
        ch = int(self.cell_h //2)
        _, bw = cv2.threshold(toByteImage(cropped[ch:-ch,cw:-cw]), \
                                0, \
                                255, \
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        def norm(x):
            return (x -x.min()) / (x.max() - x.min()) 
             
        def feat(sum):
            '''
            return the distances between the peaks
            '''
            x = (sum).astype(int)
            x_neg = -1 * (sum).astype(int)
            x = norm(x)
            x_neg = norm(x_neg)
            peaks_x, _     = find_peaks(x,     height=0.5) 
            peaks_x_neg, _ = find_peaks(x_neg, height=0.5)
            return np.diff(peaks_x), np.diff(peaks_x_neg)
 
        sum_x = np.sum(bw,0).astype(int)
        sum_y = np.sum(bw,1).astype(int)
        peaks_x, peaks_x_neg = feat(sum_x)
        peaks_y, peaks_y_neg = feat(sum_y)
         
        if np.std(peaks_x) < 1.2 and np.std(peaks_y) < 1.2:
            return True
        if np.std(peaks_x_neg) < 1.2 and np.std(peaks_y_neg) < 1.2:
            return True
        
        return False
        
        
        #only split by 17 as we removed the border fields
        px = np.array_split(sum_x, 17)
        py = np.array_split(sum_y, 17)
        idx_x = [np.argmax(x) for x in px]
        idx_y = [np.argmax(x) for x in py]

        idx_x_neg = [np.argmin(x) for x in px]
        idx_y_neg = [np.argmin(x) for x in py]

        


        if np.std(idx_x) > 1.2 or np.std(idx_y) > 1.2:
            logging.debug("std x: {}".format(np.std(idx_x)))
            logging.debug("std y: {}".format(np.std(idx_y)))
            return False
        else:
            return True


    def extract_borderless(self, img: B3CImage) -> B3CImage:
        '''
            Removes the added border around the go board
            image aligns with the boards corner lines
        '''
        img_w = self.extract(img)
        bs = self.border_size
        img_w = img_w[bs:-bs, bs:-bs]
        return img_w
 

    def extract(self, img: B3CImage) -> B3CImage:
        '''
            Returns a warped and centerd view of the go board
            with some added padding to detect hands etc.
        '''
        self.current_unwarped = img
        img_w = cv2.warpPerspective(img, np.linalg.inv(self.H), self.img_limits)
        img_c_trim, (x,y) = mask_board(img_w, self.grid, self.border_size)

        return img_c_trim

    def refine_corners(self, corners: NDArray, img : B3CImage) -> NDArray:
        '''
            Given a rough corner mask from the extraction of the largest shape on the camera 
            we can refine this mask further by matching the intersections to the refrence grid
            The previous step is neccessary as we can now extract the intersections quickly using 
            the fast Shi-Tomasi corner detector
        '''

        if corners is None:
            return None
        cH = cv2.convertPointsToHomogeneous(corners)

        # check weather we have a homography estimation
        grid = get_ref_coords(img.shape, self.border_size)

        target_corners = np.vstack((grid[342],
                                    grid[360],
                                    grid[18],
                                    grid[0]))

        H_board, _ = cv2.findHomography(target_corners, corners)
        _, (x,y) = mask_board(img, grid, self.border_size)
        go_board_shifted = grid - np.array([x,y])

        #convert corners into rectified version
        cHw = []
        for crn in cH:
            pt = np.linalg.inv(H_board) @ crn.T
            cHw.append(cv2.convertPointsFromHomogeneous(pt.T))
        cHw = np.squeeze(np.array(cHw),1).astype(int)

        limits = self.img_limits if self.img_limits is not None else (img.shape[1],img.shape[0])

        # mask everything outside the roughly found board
        H,W = limits
        mask = np.zeros((W,H), dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, cHw, 255)
        mask = cv2.bitwise_not(mask)

        # warp image and mask
        if corners is not None:
            corners = np.int32([corners])
            #cv2.polylines(img_, corners, color=(0,255,0), isClosed=True, thickness=3)

        img_ = cv2.warpPerspective(img, np.linalg.inv(H_board), limits)
        img_[mask==255] = 0


        paired_corners = self.extract_intersections(corners, img_, go_board_shifted)

        H = cv2.findHomography(go_board_shifted, paired_corners, cv2.LMEDS)[0]
        refined_H = np.linalg.inv(H) @ np.linalg.inv(H_board)
        img_warped_refined = cv2.warpPerspective(img, refined_H, limits)

        c = []
        corners__ = []
        corners__.append(go_board_shifted.reshape(19,19,2)[0,0])
        corners__.append(go_board_shifted.reshape(19,19,2)[0,18])
        corners__.append(go_board_shifted.reshape(19,19,2)[18,18])
        corners__.append(go_board_shifted.reshape(19,19,2)[18,0])


        for crn in corners__:
            crn = np.array([[crn[0], crn[1], 1]])
            pt = np.linalg.inv(refined_H) @ crn.T
            c.append(cv2.convertPointsFromHomogeneous(pt.T))
            cv2.circle(img, np.squeeze(c[-1]).astype(int), 1, (255,0,0), -1)
        c = np.squeeze(np.array(c),1).astype(int)

        #cv2.imshow('updated corners', img)
        #cv2.waitKey(1)

        # detect lines -> when the corners are correct the lines should be almost perfectly
        # vertical and horizontal
        #lines = lsd.lsd_with_line_merge(img_)
        return np.array(c)


    def mask_outside_board(self, corners: NDArray, img: B1CImage) -> B1CImage:

        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.fillPoly(mask, corners, 255)
        mask = cv2.bitwise_not(mask)
        img[mask==255] = 0
        return img

    # detect round objects
    def _mask_stones(self, corners_warped: NDArray, img: B1CImage) -> B1CImage:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(img, 11)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.bitwise_not(thresh)
        thresh = self.mask_outside_board(corners_warped, thresh)

        # Morph open 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        opening = cv2.dilate(opening, kernel)

        return opening

    def get_mask_around_stones(self, corners: NDArray, img: B3CImage) -> B1CImage:
        if len(img.shape) == 3:
            img_gray = toGrayImage(img)
            img_cmyk = toCMYKImage(img)[:,:,2]
        else:
            raise ArgumentError("Image has wrong number of channels -> we need color[rgb]")
        
        img_gray = self.mask_outside_board(corners, img_gray)
        mask_black = self._mask_stones(corners, img_gray)

        img_cmyk = self.mask_outside_board(corners, img_cmyk)
        mask_white = self._mask_stones(corners, img_cmyk)

        mask = cv2.bitwise_or(mask_black, mask_white)
        return mask

    def extract_intersections(self, corners: NDArray, img: B3CImage, go_board_shifted) -> NDArray:
        if len(img.shape) == 3:
            img__ = toGrayImage(img)
        else:
            raise ArgumentError("Image has wrong number of channels -> we need color[rgb]")
        
        stone_mask = self.get_mask_around_stones(corners, img)

        ft = cv2.goodFeaturesToTrack(img__, 19*19, 0.01, 10)
        #clean ft with mask
        idx = np.argwhere(stone_mask[ft[:,0,0].astype(int), ft[:,0,1].astype(int)] == 0)
        ft = ft[idx]

        #for c in go_board_shifted:
        #    cv2.circle(img, c.astype(int), 1, (0,0,255), 1)
        ft = np.squeeze(ft)
        dists = cdist(go_board_shifted, ft)
        paired = []
        for i, c in enumerate(go_board_shifted):
            paired.append(ft[np.argmin(dists[i])])
            #cv2.circle(img, paired[-1].astype(int), 2, (255,0,0), -1)
        #cv2.imshow('shifted', img)
        #cv2.waitKey(1)
        paired = np.array(paired)
        return paired

