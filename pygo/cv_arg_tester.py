from __future__ import print_function
from __future__ import division
from glob import iglob
from typing import Tuple, TypeVar, Callable
from pygo.GoBoard import GoBoard
from scipy.ndimage import label
import cv2 
cv2.namedWindow('bam', 0)
import matplotlib.pyplot as plt
import pdb
from pygo.utils.image import *
import numpy as np
import math
from pygo.Webcam import Webcam
from pygo.utils.typing import *
import logging
from nptyping import NDArray
from skimage.filters import sobel
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter

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

    def addArgument(self, name: str, value, range: Union[Tuple[float,float],List]):
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




class NoVanishingPointsDetectedException(Exception):
    pass



class Classifier(CV2ArgTester):
    def __init__(self, imgs):
        CV2ArgTester.__init__(self, None)
        self.imgs = imgs
        self.img_ = self.stackImgs(self.imgs)
        self.fn = self.classify
        self.params = cv2.SimpleBlobDetector_Params()
        
        self.params.minThreshold = 127
        self.params.maxThreshold = 255

        self.params.filterByCircularity = False
        self.params.minCircularity = 0.8
        self.params.maxCircularity = 1.0

        self.params.filterByConvexity = False
        self.params.minConvexity = 0.8
        self.params.maxConvexity = 1.0
        
        self.params.filterByArea = False
        self.params.minArea = 400
        self.params.maxArea = 400

        self.params.filterByInertia = False
        self.params.minInertiaRatio = 0.1
        self.params.maxInertiaRatio = 1.0
    
    def stackImgs(self, imgs):
        l = len(imgs)
        if l >= 2:
            img_a = np.vstack(imgs[:l//2])
            img_b = np.vstack(imgs[:l//2])
            img = np.hstack((img_a, img_b))
            img = cv2.resize(img, None, fx=.75, fy=.75)
        else:
            img = imgs[0]
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
        cv2.waitKey()

    def classify(self, img):

        def cmyk_prep(cmyk):
            #cmyk = cv2.medianBlur(cmyk, 5) # Add median filter to image
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
            cmyk = cv2.morphologyEx(cmyk, cv2.MORPH_ERODE, kernel, iterations=2)
            cmyk = cv2.morphologyEx(cmyk, cv2.MORPH_DILATE, kernel, iterations=2)
            return cmyk


        def img2contourbg(img):
            img_blur = cv2.blur(img, (3,3))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            #img_blur = cv2.erode(img, kernel)
            #cv2.imshow('.blur',img_blur)

            th3 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            return th3
            


        def img2contour(img):
            img_blur = cv2.blur(img, (3,3))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

            img_eq = cv2.equalizeHist(img)
            med = np.median(img_eq)
            low = int(0.66*med)
            high = int(1.33*med)
            corners = cv2.Canny(img_blur, low, high, 3)
            corners = cv2.morphologyEx(corners, cv2.MORPH_CLOSE, kernel, iterations=2)

            thresh = cv2.threshold(corners, 128, 255, cv2.THRESH_BINARY)[1]
            filled = np.zeros_like(thresh)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            for cnt in contours:
                #x,y,w,h = cv2.boundingRect(cnt)
                cv2.drawContours(filled, [cnt], 0 , 255, -1)
            return filled


        def segment_on_dt(a, img):
            border = cv2.dilate(img, None, iterations=1)
            border = border - cv2.erode(border, None)

            dt = cv2.distanceTransform(img, 2, 3)
            dt = cv2.normalize(dt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            _, dt = cv2.threshold(dt, self.get('thresh'), 1.0, cv2.THRESH_BINARY)

            lbl, ncc = label(dt)
            lbl = lbl * (255 / (ncc + 1))
            # Completing the markers now. 
            lbl[border == 255] = 255

            lbl = lbl.astype(np.int32)
            cv2.watershed(a, lbl)

            lbl[lbl == -1] = 0
            lbl = lbl.astype(np.uint8)
            return 255 - lbl

        def remove_glare(img):

            clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
            #NORMAL
            # convert to gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grayimg = gray


            GLARE_MIN = np.array([0, 0, 50],np.uint8)
            GLARE_MAX = np.array([0, 0, 225],np.uint8)

            hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

            #HSV
            frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)

            #cv2.imshow('frame_thr', frame_threshed)

            #INPAINT
            mask1 = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
            result1 = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA) 



            #CLAHE
            claheCorrecttedFrame = clahefilter.apply(grayimg)

            #COLOR 
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            #cv2.imshow('lab', lab_planes[0])

            #INPAINT + HSV
            result = cv2.inpaint(img, frame_threshed, 0.1, cv2.INPAINT_TELEA) 


            #INPAINT + CLAHE
            grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
            mask2 = cv2.threshold(grayimg1 , 220, 255, cv2.THRESH_BINARY)[1]
            result2 = cv2.inpaint(img, mask2, 0.1, cv2.INPAINT_TELEA) 

            #cv2.imshow('illu', mask2)
            #cv2.imshow('raw', img)
            #cv2.imshow('removed', result2)
            return result2

        def remove_glare2(img):
            cl = self.get("CL")
            cks = self.get("CKS")
            clahefilter = cv2.createCLAHE(clipLimit=cl,
                         tileGridSize = (cks,cks))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grayimg = gray
            
            
            GLARE_MIN = np.array([0, 0, 50],np.uint8)
            GLARE_MAX = np.array([0, 0, 225],np.uint8)
            
            hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            
            #HSV
            frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)
            
            
            #INPAINT
            mask1 = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
            result1 = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA) 
            
            
            
            #CLAHE
            claheCorrecttedFrame = clahefilter.apply(grayimg)
            
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



        def create_mask_white(ipt):
            mask_white = cv2.equalizeHist(ipt)
            mask_white[mask_white>25] = 255
            mask_white[mask_white<15] = 0
            mask_white =  cv2.adaptiveThreshold(mask_white, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 
                                            self.get("KS"), self.get('C'))
            return mask_white

        # Remove horizontal
        def remove_lines( image, thresh, orientation='h'):
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

        def binarize(img: B1CImage) -> B1CImage:
            gray = toByteImage(img.copy())
            edge = toByteImage(sobel(img))

            edge = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

            gray = remove_lines(gray, edge, 'h')
            gray = remove_lines(gray, edge, 'v')
            gray = cv2.GaussianBlur(gray,(5,5),0)
            #cv2.imshow('gray', gray)
            grid = toByteImage(sobel(gray))
            out = cv2.threshold(grid, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            # remove edges outside the boards zone which could interfere with the detection
            return out

        def detect_stones( binary_img):

            minRadius = 10
            maxRadius = 16

            hough_radii = np.arange(minRadius, maxRadius, 1)
            hough_res = hough_circle(binary_img, hough_radii)
            # Select the most prominent 3 circles
            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 
                                                    min_xdistance=minRadius,
                                                    min_ydistance=minRadius)
            circles = zip(cx,cy,radii)
            return circles

        clahe = cv2.createCLAHE(clipLimit=self.get('CL'))
        img = toColorImage(clahe.apply(toGrayImage(img)))
        cmyk = toCMYKImage(img)
        bin_img = binarize(cmyk[:,:,3])
        circles = detect_stones(bin_img)

        def plot_circles(image, circles): 
            crcl = toColorImage(image)
            for x,y,r in circles:
                circy, circx = circle_perimeter(y, x, r,
                                                shape=image.shape)
                crcl[circy, circx] = (20, 220, 20)
            return crcl

        crcl = plot_circles(bin_img.copy(), circles)
        #bin_img_inv= binarize(255-cmyk[:,:,3])
        #circles_inv = detect_stones(bin_img_inv)
        #crcl2 = plot_circles(bin_img_inv.copy(), circles_inv)
        return crcl


 
if __name__ == '__main__':
    #imgs = [cv2.imread('out{}.png'.format(i)) for i in range(6)]
    imgs = [cv2.imread('out{}.png'.format(i)) for i in [7343, 4130, 3844, 4128]]
    #ipt = Webcam()
    #board = GoBoard(ipt.getCalibration())
    #imgs = []
    #ipt.set_input_file_stream('go-spiel.mp4')
    #for i in range(6):
    #    for _ in range(5000):
    #        ipt.read()
    #    imgs.append(ipt.read())
    #    cv2.imwrite('out{}.png'.format(i), imgs[-1])
    #board.calib(imgs[2])
    #imgs = [board.extract(x) for x in imgs]
    imgs = [imgs[2]]
    argtest = Classifier(imgs)
    argtest.addArgument('med_range', 0.33, (0.0, 1.0))
    argtest.addArgument('thresh', 0.8, (0.0, 1.0))
    argtest.addArgument('C', 1, (-5, 5))
    argtest.addArgument('KS', 36, (3, 145))
    argtest.addArgument('CKS', 36, (3, 145))
    argtest.addArgument('CL', 2, (0, 20))

 
    argtest.show()
































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
        l = len(imgs)
        img_a = np.vstack(imgs[:l//2])
        img_b = np.vstack(imgs[:l//2])
        img = np.hstack((img_a, img_b))
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
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = toYUVImage(img)[:,:,0]
        #median = cv2.medianBlur(img, 5)
        #img = cv2.equalizeHist(img[:,:,0])
        #clahe = cv2.createCLAHE(self.get('cutoff'),(self.get('tile'), self.get('tile')))
        #img = clahe.apply(img)

        # threshold on black
        l = self.get('lower')
        u = self.get('upper')
        lower = (l)
        upper = (u)
        #img = cv2.inRange(img, lower, upper)
        #thresh, img = cv2.threshold(img, \
        #                            0, \
        #                            50, \
        #                            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        img = cv2.adaptiveThreshold(img,self.get("max"),cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,self.get('block'),self.get("C"))
        return img


























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