import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb

from pygo.Signals import OnBoardGridSizeKnown, Signals

class Plot:
    def __init__(self):
        self.radius = 8
        Signals.subscribe(OnBoardGridSizeKnown, self.update_radius)

    def update_radius(self, args):
        grid = args[0].reshape(19,19,2)
        dx = np.mean(np.diff(grid[:,:,0].T))
        dy = np.mean(np.diff(grid[:,:,1]))
        self.radius = int(np.mean([dx,dy])/2)



    def plot_grid(self, img, board):
        board = board.reshape(19,19,2)
        for i in range(19):
            #vertical
            start = tuple(board[0,i].astype(int))
            end   = tuple(board[18,i].astype(int))
            img = cv2.line(img, start, end, color=(255, 0, 0), thickness=1)
            #horizontal
            start = tuple(board[i,0].astype(int))
            end   = tuple(board[i,18].astype(int))
            img = cv2.line(img, start, end, color=(255, 0, 0), thickness=1)
        return img



    def plot_circles(self, img, e_cx, e_cy):
        for (x,y) in zip(e_cx, e_cy):
            img = cv2.circle(img, 
                            (int(x),int(y)), 
                            radius=self.radius, 
                            color=(255, 0, 0), 
                            thickness=1)
        return img

    def plot_circles2(self, img, c):
        for i in range(len(c)):
            img = cv2.circle(img, 
                            (int(c[i,0]),int(c[i,1])), 
                            radius=self.radius, 
                            color=(255, 0, 0), 
                            thickness=1)
        return img


    def plot_overlay(self, val, coords, img_ipt):
        img = img_ipt
        val = val.reshape(-1)
        coords = coords.reshape(-1,2)
        for v, c in zip(val, coords): 
            #if v > 0.6:
            if v == 0:
                #white
                color = (255, 255, 255)
                thickness=-1
            #elif v < 0.4:
            elif v == 1:
                #black
                color = (0,0,0)
                thickness=-1
            elif v == 2:
                color = (255, 0, 0)
                thickness=1

            img = cv2.circle(img, 
                            (int(c[0]),int(c[1])), 
                            radius=self.radius, 
                            color=color, 
                            thickness=thickness)
        return img

    def plot_virt_grid(self, val, coords, img_ipt):
        val = val.reshape(-1)
        coords = coords.reshape(-1,2)
        img = img_ipt.copy()
        for v, c in zip(val, coords): 
            if v == 0:
                #white
                color = (255, 255, 255)
                thickness=-1
                img = cv2.circle(img, 
                                (int(c[0]),int(c[1])), 
                                radius=self.radius, 
                                color=color, 
                                thickness=thickness)
                img = cv2.circle(img, 
                                 (int(c[0]),int(c[1])), 
                                 radius=self.radius, 
                                 color=(0,0,0)
                                 , thickness=1)
            elif v == 1:
                #black
                color = (0,0,0)
                thickness=-1
                img = cv2.circle(img, 
                                (int(c[0]),int(c[1])), 
                                radius=self.radius, 
                                color=color, 
                                thickness=thickness)

        return img



    def plot_val(self, val, coords, img):

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.3
        fontColor              = (255,0,0)
        lineType               = 2
        for v, c in zip(val, coords): 
            img = cv2.putText(img,
                "{:0.2f}".format(v), 
                tuple(c.astype(int)), 
                font, 
                fontScale,
                fontColor,
                lineType)
        return img

    def plot_lines(img, l_s, l_e):
        for (s,e) in zip(l_s, l_e):
            img = cv2.line(img, (int(s[0]),int(s[1])), (int(e[0]), int(e[1])), color=(255, 0, 0), thickness=1)
        return img    



    def plot_stones(self, image):

        # apply threshold
        image = color.rgb2gray(image)
        thresh_b = threshold_yen(image)
        thresh_w = threshold_minimum(1-image)
        black = closing(image > thresh_b, square(3))
        white = closing(1-image > thresh_w, square(3))
        white = np.logical_not(white)
        black = np.logical_not(black)

        # remove artifacts connected to image border
        black = clear_border(black)
        white = clear_border(white)

        # label image regions
        label_image = label(black) + label(white)
        # to make the background transparent, pass the value of `bg_label`,
        # and leave `bg_color` as `None` and `kind` as `overlay`
        image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

        plt.imshow(image_label_overlay)
        plt.show()

        image_gray = image_gray = color.rgb2gray(image)
        edges = auto_canny((image_gray*255).astype(np.uint8))
        plt.imshow(edges)
        plt.show()
        # Perform a Hough Transform
        # The accuracy corresponds to the bin size of a major axis.
        # The value is chosen in order to get a single high accumulator.
        # The threshold eliminates low accumulators
        result = hough_ellipse(edges, accuracy=20, threshold=250,
                            min_size=100, max_size=120)
        result.sort(order='accumulator')

        # Estimated parameters for the ellipse
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        image[cy, cx] = (0, 0, 255)
        plt.imshow(image)
        plt.show()