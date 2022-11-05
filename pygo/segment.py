from __future__ import print_function
from __future__ import division
import cv2 
import matplotlib.pyplot as plt
import pdb
from scipy.ndimage import label
from pygo.utils.image import *
from pygo.Webcam import Webcam
import numpy as np

class window:
    def __init__(self, img):
        self.img=img
        self.params = cv2.SimpleBlobDetector_Params()
        
        self.params.minThreshold = 127;
        self.params.maxThreshold = 255;

        self.params.filterByCircularity = True
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
        val = val -10
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
                                cv2.THRESH_BINARY_INV, self.ks, -self.c)

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
        img_ = [self.convert(img) for img in self.img]
        if len(img_) ==6:
            stacka = np.vstack(img_[:3])
            stackb = np.vstack(img_[3:])
            stack = np.hstack((stacka, stackb))
        else:
            stack = img_[0]
        cv2.imshow(title_window, stack)
    
    def convert(self, img):
        def watershed(fg):
            kernel = np.ones((3,3),np.uint8)
            fg = toByteImage(fg)

            kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

            #opening = cv2.morphologyEx(fg,cv2.MORPH_OPEN,kernel, iterations = 2)

            opening = cv2.erode(fg, kernel_cross, iterations=1)
            #opening = fg
            sure_bg = cv2.dilate(fg,kernel,iterations=3)
            cv2.imshow('opening', opening)
            cv2.imshow('sure_bg', sure_bg)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
            dist_transform = cv2.normalize(dist_transform, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            dist_color = cv2.applyColorMap(toByteImage(dist_transform), cv2.COLORMAP_JET)
            cv2.imshow('dist', dist_color)
            cv2.waitKey()

            _, dist = cv2.threshold(dist_transform, 0.8, 1.0, cv2.THRESH_BINARY)
            # Dilate a bit the dist image
            kernel1 = np.ones((3,3), dtype=np.uint8)
            dist = cv2.dilate(dist, kernel1)
            cv2.imshow('Peaks', dist)

            #ret, sure_fg = cv2.threshold(dist_transform,0.05*dist_transform.max(),255,0)
            #cv2.imshow('sure_fg', sure_fg)

            # Finding unknown region
            sure_fg = np.uint8(dist)
            unknown = cv2.subtract(sure_bg,sure_fg)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)    
            color_markers = cv2.applyColorMap(toByteImage(markers+1), cv2.COLORMAP_JET)
            cv2.imshow('markers', color_markers)
            cv2.waitKey()

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1

            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0        
            cv2.imshow('fg', fg)
            cv2.waitKey()
            markers = cv2.watershed(toColorImage(fg), markers)
            #mark = np.zeros(markers.shape, dtype=np.uint8)

            mark = markers.astype('uint8')
            mark = cv2.bitwise_not(mark)
            # uncomment this if you want to see how the mark
            # image looks like at that point
            #cv.imshow('Markers_v2', mark)
            # Generate random colors
            colors = []
            for contour in contours:
                colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
            # Create the result image
            dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
            # Fill labeled objects with random colors
            for i in range(markers.shape[0]):
                for j in range(markers.shape[1]):
                    index = markers[i,j]
                    if index > 0 and index <= len(contours):
                        dst[i,j,:] = colors[index-1]
            # Visualize the final image
            cv2.imshow('Final Result', dst)

            return markers
            

            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
            unknown = fg - sure_fg
            # Marker labelling
            cv2.imshow('fg', sure_fg)
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

        img_c = img.copy()
        hsv = toHSVImage(img.copy())
        yuv = toYUVImage(img.copy())
        cmyk = toCMYKImage(img.copy())
        img2 = toHSVImage(img.copy())[:,:,1]
        value = toGrayImage(img_c)

        #cv2.imshow('cmyk0',cmyk[:,:,0]) # good black = bright
        #cv2.imshow('cmyk1',cmyk[:,:,1]) 
        #cv2.imshow('cmyk2',cmyk[:,:,2]) 
        #cv2.imshow('cmyk3',cmyk[:,:,3])  # good black = white
        #cv2.imshow('yuv0',yuv[:,:,0]) 
        #cv2.imshow('yuv1',yuv[:,:,1]) 
        #cv2.imshow('yuv2',yuv[:,:,2]) 
        #cv2.imshow('hsv0',hsv[:,:,0]) 
        #cv2.imshow('hsv1',hsv[:,:,1]) 
        #cv2.imshow('hsv2',hsv[:,:,2]) 

        cmyk_w = cmyk[:,:,0]
        hsv = hsv[:,:,1]
        yuv = yuv[:,:,2]
        cmyk = cmyk[:,:,3]

        def cmyk_prep(cmyk):
            #cmyk = cv2.medianBlur(cmyk, 5) # Add median filter to image
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
            cmyk = cv2.morphologyEx(cmyk, cv2.MORPH_ERODE, kernel, iterations=2)
            cmyk = cv2.morphologyEx(cmyk, cv2.MORPH_DILATE, kernel, iterations=2)
            return cmyk
        #cv2.imshow('cmyk', cmyk)


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
            _, dt = cv2.threshold(dt, 0.8, 1.0, cv2.THRESH_BINARY)

            lbl, ncc = label(dt)
            lbl = lbl * (255 / (ncc + 1))
            # Completing the markers now. 
            lbl[border == 255] = 255

            lbl = lbl.astype(np.int32)
            cv2.watershed(a, lbl)

            lbl[lbl == -1] = 0
            lbl = lbl.astype(np.uint8)
            return 255 - lbl

        cmyk_b = cmyk_prep(cmyk) 
        cmyk = img2contour(cmyk_b)

        mask_w =  cv2.adaptiveThreshold(cmyk_w, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 
                                self.ks, self.c)
        cv2.imshow('',mask_w)

        filled_0 = img2contour(hsv)
        filled_1 = img2contourbg(yuv)
        diff = filled_0 - filled_1
        filled = filled_0 - diff
        mask = cv2.bitwise_and(filled, cmyk)
        mask = cv2.bitwise_or(mask, mask_w)

        markers = segment_on_dt(img_c, mask)


        # remove borders
        markers[markers==255] = 0
        markers[markers>0] = 255
        markers = 255-markers

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(self.params)
        else :
            detector = cv2.SimpleBlobDetector_create(self.params)

        keypoints = detector.detect(markers)
        markers = cv2.drawKeypoints(markers, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return markers
        im_with_kp = np.vstack((toColorImage(markers), img_c))
        cv2.imshow(title_window, im_with_kp)
        cv2.waitKey(1)
            #im_with_kp = np.vstack((toColorImage(mask_b), 
            #                        toColorImage(mask_w),
            #                        img_c))

title_window = 'Blobb'
title_window_hough = 'Hough'
#ipt = Webcam()
#ipt.set_input_file_stream('go-spiel.mp4')
#for _ in range(2380):
#    ipt.read()
#img = ipt.read()
img = cv2.imread('out20.png')
img = cv2.resize(img, None, fx=0.75, fy=0.75)
imgs = [img] 
#imgs = [cv2.imread('light0{}.png'.format(i)) for i in range(1,7)]
#imgs = [cv2.resize(img, None, fx=0.75, fy=0.75) for img in imgs]
cv2.namedWindow(title_window)
#win3 = window(img)
win = window(imgs.copy())
#cv2.namedWindow(title_window_hough)
#win2 = window(img.copy())

cv2.createTrackbar('MinThreshold', title_window , 127, 255, win.update_min_threshold)
cv2.createTrackbar('MaxThreshold', title_window , 255, 255, win.update_max_threshold)
cv2.createTrackbar('MinCirc', title_window , 75, 100, win.update_min_circ)
cv2.createTrackbar('MaxCirc', title_window , 100, 100, win.update_max_circ)
cv2.createTrackbar('minArea', title_window , 0, 500, win.update_min_area)
cv2.createTrackbar('maxArea', title_window , 0, 500, win.update_max_area)
cv2.createTrackbar('minConv', title_window , 0, 100, win.update_min_conv)
cv2.createTrackbar('maxConv', title_window , 0, 100, win.update_max_conv)
cv2.createTrackbar('minInert', title_window , 0, 100, win.update_min_inert)
cv2.createTrackbar('maxInert', title_window , 0, 100, win.update_max_inert)
cv2.createTrackbar('ks_filt', title_window , 1, 100, win.update_ks_filter)
cv2.createTrackbar('c', title_window , 10, 20, win.update_c)
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

#win.update_min_circ(10)
#win.update_max_circ(100)
#win.update_min_inert(10)
#win.update_max_inert(100)
#win.update_min_area(200)
#win.update_max_area(200)
#win.update_min_conv(0)
#win.update_max_conv(100)
win.update_ks(31)


# Wait until user press some key

cv2.waitKey()