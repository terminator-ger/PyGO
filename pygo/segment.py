from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import cv2 
import pdb
from pygo.utils.image import toCMYKImage, toByteImage, toGrayImage, toColorImage, toHSVImage
import numpy as np

class window:
    def __init__(self, img):
        self.img=img
        self.params = cv2.SimpleBlobDetector_Params()
        
        self.params.minThreshold = 127;
        self.params.maxThreshold = 255;

        self.params.filterByCircularity = True
        self.params.filterByArea = True
        self.params.minCircularity = 0.1
        self.params.maxCircularity = 1.0

        self.params.filterByConvexity = True
        self.params.minConvexity = 0.0
        self.params.maxConvexity = 1.0
        
        self.params.filterByArea = True
        self.params.minArea = 400
        self.params.maxArea = 400

        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.1
        self.params.maxInertiaRatio = 1.0

        self.minDist = 6
        self.param1 = 30 #500
        self.param2 = 20 #200 #smaller value-> more false circles
        self.minRadius = 5
        self.maxRadius = 10 #10
        self.ks = 31
        self.c = 2
        self.ks_filter = 3

    def update_min_dist(self, val):
        self.minDist = val
        self.draw_hough()

    def update_param1(self, val):
        self.param1 = val
        self.draw_hough()

    def update_param2(self, val):
        self.param2 = val
        self.draw_hough()
    
    def update_min_radius(self, val):
        self.minRadius=val
        self.draw_hough()

    def update_max_radius(self, val):
        self.maxRadius=val
        self.draw_hough()

    def update_min_threshold(self, val):
        print(val)
        self.params.minThreshold = val
        self.draw_blob()

    def update_max_threshold(self, val):
        print(val)
        self.params.maxThreshold = val
        self.draw_blob()

    def update_min_circ(self, val):
        print(val)
        self.params.minCircularity = val/100
        if self.params.minCircularity == 0 and self.params.maxCircularity == 0:
            self.params.filterByCircularity = False 
        else:
            self.params.filterByCircularity = True
        self.draw_blob()

    def update_max_circ(self, val):
        print(val/100)
        self.params.maxCircularity = val/100
        if self.params.minCircularity == 0 and self.params.maxCircularity == 0:
            self.params.filterByCircularity = False 
        else:
            self.params.filterByCircularity = True
        self.draw_blob()

    def update_min_area(self, val):
        print(val)
        self.params.minArea = val
        if self.params.minArea == 0 and self.params.maxArea == 0:
            self.params.filterByArea = False
        else:
            self.params.filterByArea = True
        self.draw_blob()

    def update_max_area(self, val):
        self.params.maxArea = val
        print(val)
        if self.params.minArea == 0 and self.params.maxArea == 0:
            self.params.filterByArea = False
        else:
            self.params.filterByArea = True
        self.draw_blob()

    def update_min_conv(self, val):
        print(val)
        self.params.minConvexity = val/100
        if self.params.minConvexity == 0 and self.params.maxConvexity == 0:
            self.params.filterByConvexity = False 
        else:
            self.params.filterByConvexity = True
        self.draw_blob()

    def update_max_conv(self, val):
        print(val)
        self.params.maxConvexity = val/100
        if self.params.maxConvexity == 0 and self.params.maxConvexity == 0:
            self.params.filterByConvexity = False 
        else:
            self.params.filterByConvexity = True
        self.draw_blob()

    def update_min_inert(self, val):
        print(val)
        self.params.minInertiaRatio = val/100
        if self.params.minInertiaRatio == 0 and self.params.maxInertiaRatio == 0:
            self.params.filterByInertia = False 
        else:
            self.params.filterByInertia = True
        self.draw_blob()

    def update_max_inert(self, val):
        print(val)
        self.params.maxInertiaRatio = val/100
        if self.params.minInertiaRatio == 0 and self.params.maxInertiaRatio == 0:
            self.params.filterByInertia = False 
        else:
            self.params.filterByInertia = True
        self.draw_blob()


    def update_c(self, val):
        #val = val-100
        self.c = val
        print(val)
        #self.draw_watershed()
        self.draw_blob()
        #self.draw_hough()
    
    def update_ks(self, val):
        if val % 2 == 0:
            val += 1
        print(val)
        self.ks = val
        #self.draw_watershed()
        self.draw_blob()
        #self.draw_hough()

    def update_ks_filter(self, val):
        if val % 2 == 0:
            val += 1
        print(val)
        self.ks_filter = val
        #self.draw_watershed()
        self.draw_blob()
        #self.draw_hough()




    def draw_watershed(self):
        img_c = self.img.copy()
        #img = toGrayImage(self.img)


        img = toHSVImage(self.img)[:,:,2]
        img = toByteImage(img)
        
        img_b = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 
                                21, 0)
                                #self.ks, self.c)
        
        img_w = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 
                                21, 0)
                                #self.ks, self.c)
        img_filt = img_w.copy()

        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (self.ks_filter, self.ks_filter))

        element_rect_v = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                            (self.ks_filter, self.ks_filter))
        element_rect_h = cv2.getStructuringElement(cv2.MORPH_RECT,
                                            (1,self.ks_filter))


        #img_b = cv2.dilate(img_b, element_rect_v) 
        #img_b = cv2.erode(img_b, element_rect_h) 
        #img_w = cv2.dilate(img_w, element) 

        img_b = cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, element)
        img_w = cv2.morphologyEx(img_w, cv2.MORPH_CLOSE, element)

        img_rect = img_w.copy()


        fg = 255-img_w
        fg = toByteImage(fg)
        dist_transform = cv2.distanceTransform(fg,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # Marker labelling
        ret, markers = cv2.connectedComponents(toByteImage(sure_fg))
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        markers = cv2.watershed(toColorImage(fg), markers)
        img_c[markers == -1] = [255,0,0]

        cv2.imshow(title_window, np.vstack((img_c, 
                                            toColorImage(img_filt),
                                            toColorImage(img_rect),)))
                                            #toColorImage(img_w), 
                                            #toColorImage(img_b))))




    def draw_hough(self):
        img = toGrayImage(self.img)
        img = toByteImage(img)
        img_w = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, self.ks, -self.c)

        img_b = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, self.ks, self.c)

        erosion_size = 1
        element = cv2.getStructuringElement(cv2.MORPH_RECT
                                        , (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    
        #retval, img_bw = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        img_w = cv2.dilate(img_w, element)
        #img_b = cv2.dilate(img_b, element)

        circles_w = cv2.HoughCircles(img_w,
                    cv2.HOUGH_GRADIENT, 1, 
                    self.minDist, 
                    param1=self.param1, 
                    param2=self.param2, 
                    minRadius=self.minRadius, 
                    maxRadius=self.maxRadius)
        circles_b = cv2.HoughCircles(img_b,
                    cv2.HOUGH_GRADIENT, 1, 
                    self.minDist, 
                    param1=self.param1, 
                    param2=self.param2, 
                    minRadius=self.minRadius, 
                    maxRadius=self.maxRadius)
               
        def draw(circles, color, img):
            if circles is not None:
                circles = circles.astype(np.int)
                for i in circles[0,:]:
                    cv2.circle(img, (i[0], i[1]), i[2], color, 1)
        draw(circles_b, (255,0,0), toColorImage(img_w))
        draw(circles_w, (0,255,0), toColorImage(img_b))
        #params.maxArea = 100;

 
        cv2.imshow(title_window_hough, np.vstack((img_b, img_w)))



    def draw_blob(self):
        img_c = self.img.copy()
        img = toHSVImage(self.img.copy())[:,:,2]
        img2 = toHSVImage(self.img.copy())[:,:,1]
        img = toByteImage(img)
        img2 = toByteImage(img2)
        
        img_b = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 
                                #21, 0)
                                self.ks, self.c)
        
        img_w = cv2.adaptiveThreshold(img2, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 
                                #21, 0)
                                self.ks, -self.c)

        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (self.ks_filter, self.ks_filter))

        #img_b = cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, element)
        #img_w = cv2.morphologyEx(img_w, cv2.MORPH_CLOSE, element)

        def watershed(fg):
            fg = toByteImage(fg)
            dist_transform = cv2.distanceTransform(fg,cv2.DIST_L2,3)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
            unknown = fg - sure_fg
            # Marker labelling
            ret, markers = cv2.connectedComponents(toByteImage(sure_fg))
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1
            markers[unknown.astype(bool)]  = 0
            markers = cv2.watershed(toColorImage(fg), markers)
            markers[markers==1] = 0
            markers[markers>1] = 255
            markers[markers==-1] = 0
            markers = toByteImage(markers)
            markers = cv2.erode(markers, 
                                cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
            markers = cv2.GaussianBlur(toGrayImage(markers), (3,3), 2)
            return markers
        mask = 255-watershed(cv2.bitwise_or(cv2.bitwise_not(img_b), 
                                            cv2.bitwise_not(img_w)))

        mask_b = 255-watershed(255-img_b)
        mask_w = 255-watershed(255-img_w)
        
        img_masks = [mask_b, mask_w]

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(self.params)
        else :
            detector = cv2.SimpleBlobDetector_create(self.params)

        for i in img_masks:
            keypoints = detector.detect(i)
            img_c = cv2.drawKeypoints(img_c, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            im_with_kp = np.vstack((toColorImage(mask_b), 
                                    toColorImage(mask_w),
                                    img_c))
        cv2.imshow(title_window, im_with_kp)

title_window = 'Blobb'
title_window_hough = 'Hough'
img = cv2.imread('out.png')
cv2.namedWindow(title_window)
#win3 = window(img)
win = window(img.copy())
#cv2.namedWindow(title_window_hough)
#win2 = window(img.copy())

cv2.createTrackbar('MinThreshold', title_window , 127, 255, win.update_min_threshold)
cv2.createTrackbar('MaxThreshold', title_window , 255, 255, win.update_max_threshold)
cv2.createTrackbar('MinCirc', title_window , 81, 100, win.update_min_circ)
cv2.createTrackbar('MaxCirc', title_window , 100, 100, win.update_max_circ)
cv2.createTrackbar('minArea', title_window , 200, 500, win.update_min_area)
cv2.createTrackbar('maxArea', title_window , 500, 500, win.update_max_area)
cv2.createTrackbar('minConv', title_window , 50, 100, win.update_min_conv)
cv2.createTrackbar('maxConv', title_window , 60, 100, win.update_max_conv)
cv2.createTrackbar('minInert', title_window , 60, 100, win.update_min_inert)
cv2.createTrackbar('maxInert', title_window , 60, 100, win.update_max_inert)
cv2.createTrackbar('ks_filt', title_window , 1, 100, win.update_ks_filter)
cv2.createTrackbar('c', title_window , 2, 20, win.update_c)
cv2.createTrackbar('ks', title_window , 21, 200, win.update_ks)

#cv2.createTrackbar('minDist', title_window_hough , 20, 100, win2.update_min_dist)
#cv2.createTrackbar('minRadius', title_window_hough , 5, 50, win2.update_min_radius)
#cv2.createTrackbar('maxRadius', title_window_hough , 15, 50, win2.update_max_radius)
#cv2.createTrackbar('param1', title_window_hough , 15, 200, win2.update_param1)
#cv2.createTrackbar('param2', title_window_hough , 15, 200, win2.update_param2)

#cv2.createTrackbar('ks', title_window , 31, 200, win.update_ks)
#cv2.createTrackbar('c', title_window , 102, 200, win.update_c)

#cv2.createTrackbar('ks', title_window , 42, 200, win3.update_ks)
#cv2.createTrackbar('ks_filter', title_window , 42, 200, win3.update_ks_filter)
#cv2.createTrackbar('c', title_window , 103, 200, win3.update_c)

#cv2.createTrackbar('ks', title_window_hough , 21, 200, win2.update_ks)
#cv2.createTrackbar('c', title_window_hough , 100, 200, win2.update_c)
# Show some stuff

win.update_min_circ(10)
win.update_max_circ(100)
win.update_min_inert(10)
win.update_max_inert(100)
win.update_min_area(200)
win.update_max_area(200)
win.update_min_conv(0)
win.update_max_conv(100)
win.update_ks(31)


# Wait until user press some key

cv2.waitKey()