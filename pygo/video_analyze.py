from __future__ import division 
from sklearn.mixture import GaussianMixture
from pylab import concatenate, normal
import scipy.stats as stats
import pdb
import random
from numpy.core.numeric import ones
from tqdm import tqdm
import cv2
from icp import icp
import numpy as np
from pyELSD.pyELSD import PyELSD
from lu_vp_detect import VPDetection
import matplotlib.pyplot as plt
from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration
from functools import partial
from scipy import ndimage
from skimage.exposure import equalize_hist
from joblib import load, dump
from sklearn.mixture import GaussianMixture
from multiprocessing.connection import Client
from pylab import *
from scipy.optimize import curve_fit


from skimage import data, color, img_as_ubyte, exposure, transform, img_as_float
from skimage.feature import canny, hog, corner_harris, corner_subpix, corner_peaks
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import collections
from skimage import data
from skimage.filters import threshold_yen, threshold_minimum
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from scipy.spatial import distance_matrix
from sklearn.neighbors import KNeighborsClassifier
import torch as th
from numpy.lib.shape_base import expand_dims
import sklearn
import cv2
from datetime import datetime
from enum import Enum
import os
import numpy as np
from skimage import data, color, img_as_ubyte, exposure, transform, img_as_float
from skimage.feature import canny, hog, corner_harris, corner_subpix, corner_peaks
from skimage import data
from skimage.measure import label, regionprops
import pdb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from joblib import dump, load
import imgaug as ia
import imgaug.augmenters as iaa
import torch as th
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from sgfmill import sgf
from scipy.stats import skew, moment
from skimage.feature import hog
import numpy as np
from skimage import exposure, img_as_float
import pdb
from sklearn.linear_model import SGDClassifier
import sys
from time import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from classifier import GoClassifier

from dask import delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

from pygo_utils import toByteImage

import sys
sys.path.insert(0, './train')


import warnings
warnings.filterwarnings('always') 

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def warp_lines(lines, H):
    ones = np.ones((lines.shape[0], 1))
    l_homo = np.concatenate((lines, ones), axis=1)
    l_warped = l_homo @ H
    return l_warped

def compute_homography_and_warp(image, vp1, vp2, clip=True, clip_factor=3):
    """Compute homography from vanishing points and warp the image.
    It is assumed that vp1 and vp2 correspond to horizontal and vertical
    directions, although the order is not assumed.
    Firstly, projective transform is computed to make the vanishing points go
    to infinty so that we have a fronto parellel view. Then,Computes affine
    transfom  to make axes corresponding to vanishing points orthogonal.
    Finally, Image is translated so that the image is not missed. Note that
    this image can be very large. `clip` is provided to deal with this.
    Parameters
    ----------
    image: ndarray
    Image which has to be wrapped.
    vp1: ndarray of shape (3, )
    First vanishing point in homogenous coordinate system.
    vp2: ndarray of shape (3, )
    Second vanishing point in homogenous coordinate system.
    clip: bool, optional
    If True, image is clipped to clip_factor.
    clip_factor: float, optional
    Proportion of image in multiples of image size to be retained if gone
    out of bounds after homography.
    Returns
    -------
    warped_img: ndarray
    Image warped using homography as described above.
    """
    # Find Projective Transform
    vanishing_line = np.cross(vp1, vp2)
    H = np.eye(3)
    H[2] = vanishing_line / vanishing_line[2]
    H = H / H[2, 2]

    # Find directions corresponding to vanishing points
    v_post1 = np.dot(H, vp1)
    v_post2 = np.dot(H, vp2)
    v_post1 = v_post1 / np.sqrt(v_post1[0]**2 + v_post1[1]**2)
    v_post2 = v_post2 / np.sqrt(v_post2[0]**2 + v_post2[1]**2)

    directions = np.array([[v_post1[0], -v_post1[0], v_post2[0], -v_post2[0]],
    [v_post1[1], -v_post1[1], v_post2[1], -v_post2[1]]])

    thetas = np.arctan2(directions[0], directions[1])

    # Find direction closest to horizontal axis
    h_ind = np.argmin(np.abs(thetas))

    # Find positve angle among the rest for the vertical axis
    if h_ind // 2 == 0:
        v_ind = 2 + np.argmax([thetas[2], thetas[3]])
    else:
        v_ind = np.argmax([thetas[2], thetas[3]])

    A1 = np.array([[directions[0, v_ind], directions[0, h_ind], 0],
    [directions[1, v_ind], directions[1, h_ind], 0],
    [0, 0, 1]])
    # Might be a reflection. If so, remove reflection.
    if np.linalg.det(A1) < 0:
        A1[:, 0] = -A1[:, 0]

    A = np.linalg.inv(A1)

    # Translate so that whole of the image is covered
    inter_matrix = np.dot(A, H)

    cords = np.dot(inter_matrix, [[0, 0, image.shape[1], image.shape[1]],
    [0, image.shape[0], 0, image.shape[0]],
    [1, 1, 1, 1]])
    cords = cords[:2] / cords[2]

    tx = min(0, cords[0].min())
    ty = min(0, cords[1].min())

    max_x = cords[0].max() - tx
    max_y = cords[1].max() - ty

    if clip:
        # These might be too large. Clip them.
        max_offset = max(image.shape) * clip_factor / 2
        tx = max(tx, -max_offset)
        ty = max(ty, -max_offset)

        max_x = min(max_x, -tx + max_offset)
        max_y = min(max_y, -ty + max_offset)

    max_x = int(max_x)
    max_y = int(max_y)

    T = np.array([[1, 0, -tx],
    [0, 1, -ty],
    [0, 0, 1]])

    final_homography = np.dot(T, inter_matrix)

    warped_img = cv2.warpPerspective(image, (final_homography), (max_x, max_y))
    return warped_img, final_homography, (max_x, max_y)

def filter_vp(vp, img_size=np.array([740,420])):
    """
    :param vp: 2x3 vector with vp coordinates
    :return: selected vp for rectification
    """
    vp_copy = vp.copy()
    selected_vp = []
    # shift to center
    vp = vp - img_size/2
    # calc angle
    angle = np.arctan2(vp[:,1],vp[:,0])
    threshhold = np.deg2rad(35)
    is_vertical = np.logical_or(np.abs(angle - np.pi / 2) < threshhold,
    np.abs(angle + np.pi / 2) < threshhold)
    dist = np.sqrt(np.sum(np.square(vp),axis=1))
    dist[np.logical_not(is_vertical)] = 0
    vert_IDX = np.argmax(dist)

    vp1 = np.array([vp_copy[vert_IDX,0],vp_copy[vert_IDX,1],1])
    # set zero
    vp[vert_IDX] = 0

    dist = np.sqrt(np.sum(np.square(vp),axis=1))
    horiz_IDX = np.argmax(dist)
    vp2 = np.array([vp_copy[horiz_IDX,0],vp_copy[horiz_IDX,1],1])
    return vp1, vp2

def plot_grid(img, board):
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

def plot_circles(img, e_cx, e_cy):
    for (x,y) in zip(e_cx, e_cy):
        img = cv2.circle(img, (int(x),int(y)), radius=5, color=(255, 0, 0), thickness=1)
    return img

def plot_circles2(img, c):
    for i in range(len(c)):
        img = cv2.circle(img, (int(c[i,0]),int(c[i,1])), radius=5, color=(255, 0, 0), thickness=1)
    return img


def clear_bimodal(lines, image_height):
    circ_dists = distance_matrix(lines, lines, 2)
    dists = circ_dists
    # sort by closes eight dists
    N = 8
    circ_dists_idx = np.argsort(circ_dists, axis=0)
    circ_dists = np.sort(circ_dists, axis=0)
    circ_dists = circ_dists[1:(1+N)]
    # assumption is that the whole board is visible so the the max distances are w/19, h/19 and the board fills at least half the images height..
    max_dist = image_height /19
    min_dist = image_height/2 /19
    circ_dists = circ_dists.reshape(-1,1)
    circ_dists = np.delete(circ_dists, np.argwhere(circ_dists>max_dist))
    circ_dists = np.delete(circ_dists, np.argwhere(circ_dists<min_dist))

    f = np.ravel(circ_dists).astype(np.float)
    f=f.reshape(-1,1)
    # Determine parameters mu1, mu2, sigma1, sigma2, w1 and w2
    gm = GaussianMixture(n_components=4, random_state=0).fit(f.reshape(-1,1))

    mu = gm.means_
    std = (gm.covariances_)
    weights = gm.weights_
    # discard wide distribution
    for i in range(2):
        delidx= np.argmin(weights)
        mu  = np.delete(mu,  delidx)
        std = np.delete(std, delidx)
        weights   = np.delete(weights,   delidx)
    #plt.hist(f, bins=100, histtype='bar', density=True, ec='red', alpha=0.5)

    f_axis = f.copy().ravel()
    f_axis.sort()
    plt.hist(f, bins=100, histtype='bar', density=True, ec='red', alpha=0.5)
    plt.plot(f_axis,weights[0]*stats.norm.pdf(f_axis,mu[0],np.sqrt(std[0])).ravel(), c='red')
    plt.plot(f_axis,weights[1]*stats.norm.pdf(f_axis,mu[1],np.sqrt(std[1])).ravel(), c='red')
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.grid()
    plt.show()
    # get index of matching points
    idx0 = np.argwhere(np.abs(dists - mu[0]) < 1.2* std[0])
    idx1 = np.argwhere(np.abs(dists - mu[1]) < 1.2* std[1])
    idx = np.intersect1d(idx0,idx1)
    lines_selected = lines[idx]

    return lines_selected

def fit_gmm_1d(x, N=1, F=1., top=1):
    x = np.ravel(x).astype(np.float)
    x=x.reshape(-1,1)
    # Determine parameters mu1, mu2, sigma1, sigma2, w1 and w2
    gm = GaussianMixture(n_components=N, random_state=0).fit(x)
    mu = gm.means_.tolist()
    sig = gm.covariances_.tolist()
    w = gm.weights_.tolist()
    MU = []
    SIG = []
    W = []
    for i in range(top):
        idx = np.argmax(w)
        MU.append(mu.pop(idx))
        W.append(sig.pop(idx))
        SIG.append(w.pop(idx))

    idx = [np.argwhere(np.abs(x - MU[i]) < F * SIG[i]) for i in range(top)]
    return idx 
 
 

def clear(lines):
    # cluster x
    from scipy.spatial.distance import pdist
    from scipy.spatial import  distance_matrix
    from sklearn.cluster import KMeans, AgglomerativeClustering
    circ_dists = distance_matrix(lines, lines, 2)
    x_data = lines[:,0]
    # sort by closes two dists
    N = 8
    circ_dists = np.sort(circ_dists, axis=0)
    circ_dists_idx = np.argsort(circ_dists, axis=0)
    circ_dists = circ_dists[1:(1+N)]
    pdb.set_trace()
    med_dist = np.median(circ_dists)
    circ_dists = np.square(circ_dists-med_dist)

    circ_dists_idx = circ_dists_idx[1:(1+N)]
    km = AgglomerativeClustering(n_clusters=None, 
                                linkage='single',
                                distance_threshold=4)
    lblx = km.fit_predict(circ_dists.T)
    plt.scatter(lines[:,0], lines[:,1], c=lblx)
    plt.show()
    pdb.set_trace()

    _, xcount = np.unique(lblx, return_counts=True)
    target_cluster = np.argmax(xcount)
    delx = np.argwhere(lblx != target_cluster)
    lines = np.delete(lines, delx, axis=0)
    lblx = np.delete(lblx, delx)

    plt.scatter(lines[:,0], lines[:,1], c=lblx)
    plt.show()
    pdb.set_trace()
   # x_data = lines[:,1]
   # lblx = km.fit_predict(x_data.reshape(-1,1))
   # #plt.scatter(lines[:,0], lines[:,1], c=lblx)
   # #plt.show()
   # _, xcount = np.unique(lblx, return_counts=True)
   # delx = np.where(lblx == np.argwhere(xcount < 10))[1]
   # plt.scatter(lines[:,0], lines[:,1], c=lblx)
   # plt.show()
 
   # lines = np.delete(lines, delx, axis=1)
 
    return lines

def plot_lines(img, l_s, l_e):
    for (s,e) in zip(l_s, l_e):
        img = cv2.line(img, (int(s[0]),int(s[1])), (int(e[0]), int(e[1])), color=(255, 0, 0), thickness=1)
    return img    

def lines_to_2d(l_s, l_e):
    L = len(l_s)//2
    lsx = np.zeros(L)
    lsy = np.zeros(L)
    lex = np.zeros(L)
    ley = np.zeros(L)        
    idx=0

    for i in range(0,len(l_s),2):
        lsx[idx] = l_s[i]
        lsy[idx] = l_s[i+1]

        lex[idx] = l_e[i]
        ley[idx] = l_e[i+1]
        idx += 1
    ls = np.stack((lsx, lsy)).T
    le = np.stack((lex, ley)).T
    return ls, le

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def intersect(x,y):
    l1 = []
    l2 = []
    for i in range(len(x)):
        l1.append(line(x[i,:2], x[i,2:]))

    for i in range(len(y)):
        l2.append(line(y[i,:2], y[i,2:]))
    i = []
    for p1 in l1:
        for p2 in l2:
            i.append(intersection(p1,p2))
        
    return np.array(i)

def cluster(x,y):
    '''
    receives vertial and horizontal lines (belonging to a vanishing point)
    of the shape (N,4) with [x,y,x,y] being the coords of each lines (start,end)
    '''
    from scipy.cluster.hierarchy import linkage, average, fcluster
    x = np.squeeze(np.array(x))
    y = np.squeeze(np.array(y))
    arr = np.concatenate((x,y)).reshape(-1,2)
    X = average(arr.reshape(-1,2))
    cl = fcluster(X, 10, criterion='distance')
    #c = []
    #for i in range(1, cl.max()):
    #c.append(np.mean(arr[np.argwhere(cl==i),:], axis=0))
    c = [np.mean(arr[np.argwhere(cl==i),:], axis=0) for i in range(1,cl.max())]
    c = np.array(c).reshape(-1,2)
    #plt.subplot(211)
    #plt.scatter(arr[:,0],arr[:,1], label='orig')
    #plt.subplot(212)
    #plt.scatter(c[:,0],c[:,1], label='cleaned')
    #plt.show()
    return c

def get_ref_go_board_coords(min, max):
    # assume symmectric go board
    dpx = (max[0]-min[0]) / 19
    dpy = (max[1]-min[1]) / 19
    go_board = np.zeros((19, 19, 2))

    for i in range(19):
        for j in range(19):
            go_board[i, j, 0] = i * dpx 
            go_board[i, j, 1] = j * dpy 

    go_board += np.array(min)
    return go_board.reshape(-1,2)

def initial_point_scaling(src, dst):
    '''
        src, dst [Nx2]
    '''
    x = np.mean(dst[:,0])
    y = np.mean(dst[:,1])
    sx = np.max(dst[:,0]) - np.min(dst[:,0])
    sy = np.max(dst[:,1]) - np.min(dst[:,1])

    src_x = np.mean(src[:,0])
    src_y = np.mean(src[:,1])

    src_sx = np.max(src[:,0]) - np.min(src[:,0])
    src_sy = np.max(src[:,1]) - np.min(src[:,1])

    dx = src_x - x
    dy = src_y - y
    scale_x = sx/src_sx
    scale_y = sy/src_sy
    T = np.eye(3)
    T[0,0] = scale_x
    T[1,1] = scale_y
    T[0,2] = dx
    T[1,2] = dy
    return T

def crop(pts, img):
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

def plot_val(val, coords, img):
    
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

def plot_overlay(val, coords, img):
    val = val.reshape(-1)
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

        img = cv2.circle(img, (int(c[0]),int(c[1])), radius=8, color=color, thickness=thickness)
    return img

def plot_stones(image):
    
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

def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def scplt(x, c=None):
            if c is not None:
                plt.scatter(x[:,0], x[:,1], c=c)
            else:
                plt.scatter(x[:,0], x[:,1])


def mask_board(image, go_board):
    mask = np.zeros_like(image)
    go = go_board.reshape(19,19,2)
    corners = np.array([go[0,0], 
                        go[0,18], 
                        go[18,18], 
                        go[18,0]]).reshape(-1,1,2).astype(int)
    mask = cv2.fillConvexPoly(mask, corners, (255,255,255))
    mask = cv2.dilate(mask, np.ones((3,3)), iterations=15)
    if len(mask.shape) == 3:
        rect = cv2.boundingRect(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))               # function that computes the rectangle of interest
    else:
        rect = cv2.boundingRect(mask.astype(np.uint8))
    x,y,w,h = rect
    cropped_img = image[y:y+h, x:x+w].copy()
    return cropped_img, (x,y)

def find_src_pt(go, lines):
    # find nn
    circ_dists = distance_matrix(go, lines, 2)
    nn_idx = np.argmin(circ_dists, axis=1)
    nn = lines[nn_idx,:]
    # remove unmatched
    return nn

class CameraCalib:
    def __init__(self, intr) -> None:
        self.focal = intr[0,0]
        self.intr = intr
        self.center = (intr[0,2], intr[1,2])

class GoBoard:
    def __init__(self, CameraCalib):
        self.H = np.eye(3)
        self.vp = VPDetection(focal_length=CameraCalib.focal, 
                              principal_point=CameraCalib.center, 
                              length_thresh=10)
        self.grid = None
        self.go_board_shifted = None
        self.hasEstimate = False

    def crop(self, pts, img):
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

    def imgToPatches(self, img):
        patches = []
        for path in zip(self.cl2, self.ct2, self.cr2, self.cb2):
            #l,r,t,b):
            l1 = np.array([path[0][0], path[1][1]])
            l2 = np.array([path[2][0], path[1][1]])
            l3 = np.array([path[2][0], path[3][1]])
            l4 = np.array([path[0][0], path[3][1]])
            p = np.array([l1,l2,l3,l4]).astype(int)
        # patch = cv2.cvtColor(crop(p, img), cv2.COLOR_BGR2GRAY)
            patch = crop(p, img)
        #        patch = patch / 255.0

            patch =  transform.resize(patch, (32,32),  anti_aliasing=True)
            patches.append(patch)
            #m = np.nanmean(patch)
        return patches

    def linesToLength(self, lines):
        '''lines [N,4]'''
        dx = lines[:,0] - lines[:,2]
        dy = lines[:,1] - lines[:,3]
        d = np.sqrt(dx**2 + dy**2)
        return d

    def filterByLength(self, lines):
        length = self.linesToLength(lines)
        idx = fit_gmm_1d(length, N=2, F=1.5)
        return lines[idx]


    def calib(self, img):
        h,w = img.shape
        van_points = self.vp.find_vps(img)
        vpd = self.vp.create_debug_VP_image()
        cv2.imshow('PyGO', vpd)
        cv2.waitKey(500)
        lines_vp = []
        vps = self.vp.vps_2D
        vp3d = self.vp.vps
        vert_vp = np.argmax(np.abs(vp3d[:,2]))
        vps = np.delete(vps, vert_vp, axis=0)
        vp1 = np.array([vps[0,0], vps[0,1], 1])
        vp2 = np.array([vps[1,0], vps[1,1], 1])
        vps_list_1.append(vp1)
        vps_list_2.append(vp2)        
        lines_cluster = self.vp.get_lines()
        # drop lines with least 
        del lines_cluster[(np.argmin([len(x) for x in lines_cluster]))]
        lines_vp.append(lines_cluster)

        vp1 = np.median(np.array(vps_list_1), axis=0)
        vp2 = np.median(np.array(vps_list_2), axis=0)
        warped_img, H, img_limits = compute_homography_and_warp(img, vp1, vp2, clip=False, clip_factor=1)
        self.img_limits = img_limits
        img_min = (img.shape[1],img.shape[0])
        img_max = (0,0)


        img_cw = cv2.warpPerspective(img, H, img_limits)
        img_dcw = np.asarray(img_cw, dtype=np.double, order='C')

        lines_warpedV = []
        lines_warpedH = []

        for lines in lines_vp:
            lines_warpedV.append(cv2.perspectiveTransform(np.stack(lines[0]).reshape(-1,1,2), H).reshape(-1,4))
            lines_warpedH.append(cv2.perspectiveTransform(np.stack(lines[1]).reshape(-1,1,2), H).reshape(-1,4))
        # filter lines by section length

        self.lines_warpedV = self.filterByLength(lines_warpedV[0])
        self.lines_warpedH = self.filterByLength(lines_warpedH[0])
        

        lines_raw_orig = cluster(lines[0], lines[1])
        lines = cluster(lines_warpedH, lines_warpedV)
        h,w = img_cw.shape
        lines = clear_bimodal(lines, h)
        #plt.title("after gmm")
        #plt.imshow(img_cw)
        #plt.scatter(lines[:,0], lines[:,1])
        #plt.show()

        img_max = (img_cw.shape[1], img_cw.shape[0])
        img_min = (0,0)

        self.grid = get_ref_go_board_coords(np.min(lines, axis=0), np.max(lines, axis=0))
        img_cw = cv2.warpPerspective(img, H, img_limits)

        def visualize(iteration, error, X, Y, ax):
            if iteration % 1 == 0:
                plt.cla()
                ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
                ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
                plt.imshow(img)
                plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
                    iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
                ax.legend(loc='upper left', fontsize='x-large')
                plt.draw()
                plt.pause(0.001)


        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        callback = partial(visualize, ax=fig.axes[0])
        grid17x17 = self.grid.reshape(19,19,2)[1:-1,1:-1].reshape(-1,1,2)
        lines       = np.squeeze(cv2.perspectiveTransform(lines.reshape(-1,1,2),     np.linalg.inv(H), (img.shape[1], img.shape[0]))).squeeze()
        grid17x17   = np.squeeze(cv2.perspectiveTransform(grid17x17,                 np.linalg.inv(H), (img.shape[1], img.shape[0]))).squeeze()
        self.gridW  = np.squeeze(cv2.perspectiveTransform(self.grid.reshape(-1,1,2), np.linalg.inv(H), (img.shape[1], img.shape[0]))).squeeze()

        reg = AffineRegistration(**{'Y': lines, 
                                    'X': grid17x17, 
                                #'max_iterations': 15000,
                                #'tolerance': 0.0000001,
                                #'B' : np.eye(2)*5
                                })
        #ty, param = reg.register(callback)
        ty, param = reg.register()
        Rot   = param[0]
        Trans = param[1]
        RotI = np.linalg.inv(Rot)

        #scplt(grid17x17)
        #wl = np.dot(lines, Rot) + np.tile(Trans, (lines.shape[0], 1))
        #scplt(wl)
        #plt.show()

        #scplt(lines)
        #wg =  np.dot(grid17x17 - np.tile(Trans, (grid17x17.shape[0],1)), np.linalg.inv(Rot))
        #scplt(wg)
        #plt.show()
        self.gridW = self.gridW.reshape(-1,2) 
        ow_board = (self.gridW - np.tile(Trans, (self.gridW.shape[0],1))) @ RotI

        T = np.eye(3)
        T[:2,2] = Trans
        RotIH = np.eye(3)
        RotIH[:2,:2] = RotI
        H_ = (H @ T) @ RotIH
        #print(H)
        #print(H_)
        #plt.imshow(cv2.warpPerspective(img, H_, img_limits))
        #plt.show()

        #self.H = (R) @ np.linalg.inv(H)
        #self.H =  np.linalg.inv(H)
        self.H =  H_
        self.hasEstimate = True
        img_grid = plot_grid(img.copy(), ow_board.reshape(-1,2))
        cv2.imshow('PyGO', img_grid)
        cv2.waitKey(1500)       
        Trans = np.concatenate((np.eye(2), Trans.reshape(2,1)),axis=1)
        Rot  = np.concatenate((RotI, np.zeros((2,1))),axis=1)
        imgW = cv2.warpPerspective(img, (H), self.img_limits)
        imgW = cv2.warpAffine(imgW,  Trans, self.img_limits)
        imgW = cv2.warpAffine(imgW, Rot, self.img_limits)
        plt.imshow(imgW)
        scplt(self.grid.reshape(-1,2))
        plt.show()
        pdb.set_trace()
        #plt.imshow(img_cw)
        #plt.scatter(w_board[:,0], w_board[:,1])
        #plt.show()

        #img_grid = plot_grid(img.copy(), ow_board.reshape(-1,2))
        #cv2.imshow('PyGO', img_grid)
        #cv2.waitKey(500)        
    
        #src_pt = find_src_pt(ow_board, lines_raw_orig)
        #pdb.set_trace()
        #H_refined = cv2.findHomography(src_pt, ow_board)[0]
        #w_board = cv2.perspectiveTransform(np.array([self.grid]), R @ np.linalg.inv(H) @ np.linalg.inv(H_refined))[0]

        #H_refined = R @ np.linalg.inv(H) @ np.linalg.inv(H_refined)

        #img_grid = plot_grid(img.copy(), w_board.reshape(-1,2))
        #cv2.imshow('PyGO', img_grid)
        #cv2.waitKey(1500)       
        #self.H = H_refined 
        #self.hasEstimate=True

        #print(self.H)
 
    def extract(self, img):
        img_w = cv2.warpPerspective(img, np.linalg.inv(self.H), self.img_limits)

        img_c_trim, (x,y) = mask_board(img_w, self.grid)
        self.go_board_shifted = self.grid - np.array([x,y])

        #determined spaces from grid spacing
        cell_w = np.mean(np.diff(self.go_board_shifted.reshape(19,19,2)[:,:,0], axis=0))
        cell_h = np.mean(np.diff(self.go_board_shifted.reshape(19,19,2)[:,:,1], axis=1))
        
        self.cl2 = self.go_board_shifted - np.array([cell_w/2, 0])
        self.cr2 = self.go_board_shifted + np.array([cell_w/2, 0])
        self.ct2 = self.go_board_shifted + np.array([0, cell_h/2])
        self.cb2 = self.go_board_shifted - np.array([0, cell_h/2])

        return img_c_trim

    def extractOnPoints(self, img):
        img = self.extract(img)
        points_w = np.array([[ 4, 3],
                             [10, 3],
                             [16, 3],
                             [ 4, 9],
                             [10, 9],
                             [16, 9],
                             [ 4,15],
                             [10,15],
                             [16,15]])
        points_b = np.array([[ 2 ,3],
                             [ 8 ,3],
                             [14 ,3],
                             [ 2 ,9],
                             [ 8 ,9],
                             [14 ,9],
                             [ 2,15],
                             [ 8,15],
                             [14,15]])

        idx_w = np.ravel_multi_index(points_w.T, (19,19))
        idx_b = np.ravel_multi_index(points_b.T, (19,19))
        idx_n = np.arange(19*19)
        idx_n = np.delete(idx_n, np.concatenate((idx_w,idx_b)))
        patches = []
        p = self.imgToPatches(img)
        for idx in [idx_w, idx_b, idx_n]:
            patches.append(np.array(p)[idx])
        return patches

class GameState(Enum):
    RUNNING = 0
    NOT_STARTED = 1

class Game:
    def __init__(self):
        self.state = np.ones((19,19),dtype=np.int64)*2
        self.last_color = 2
        self.last_x = -1
        self.last_y = -1
        # 0 = white
        # 1 = black
        # 2 = empty
        self.GS = GameState.NOT_STARTED
        self.sgf = None
        self.sgf_node = None
       
    def startNewGame(self, size=19):
        self.sgf = sgf.Sgf_game(size=size)
        self.sgf_node = self.sgf.get_root()
        self.GS = GameState.RUNNING
       
    def endGame(self):
        if self.GS == GameState.RUNNING:
            cur_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
            with open('{}.sgf'.format(cur_time), 'wb') as f:
                f.write(self.sgf.serialise())

            self.GS = GameState.NOT_STARTED
            self.sgf = None

    def updateState(self, state):
        state = state.reshape(19,19)
        diff = np.count_nonzero(np.abs(self.state-state))
        if  diff > 1:
            # check for removed stones
            idx = np.argwhere(np.abs(self.state-state)>0)
            X = idx[:,0]
            Y = idx[:,1]
            isInTree = []
            notInTree = []
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                c = state[x,y]
                c_str = N2C(c)
                if self.GS == GameState.RUNNING:
                    seq = self.sgf.get_main_sequence()[:-2]
                    for node in seq:
                        if node.properties()[-1] == c_str and (x,y) == node.get(c_str):
                            isInTree.append((c_str,(x,y)))
                            break
                    # not found
                    notInTree.append((c_str,(x,y)))
            if len(notInTree) == 1 and notInTree[0][0] != self.last_color:
                # stones where captured
                c_str = notInTree[0][0]
                (x,y) = notInTree[0][1]
                self.sgf_node.set_move(c_str.lower(),(x,y))
                self.sgf_node = self.sgf.extend_main_sequence()
                self.last_x = x
                self.last_y = y
                print('{}: {}-{}'.format(c_str, x+1, y+1))
                self.state = state
                self.last_color = c
            # more than one move changed
            return
        # get last moves color
        idx = np.argwhere(np.abs(self.state-state)>0)
        if idx.size==0:
            return
        x = idx[0,0]
        y = idx[0,1]
        c = state[x,y]

        if c != self.last_color:
            c_str = N2C(c)
            if self.GS == GameState.RUNNING:
                #check wether the move is the last in the tree:
                seq = self.sgf.get_main_sequence()
                if c_str == 'E' and len(seq) > 0:
                    last = seq[-2]
                    last_move_color = last.properties()
                    last_move_pos = last.get(last_move_color[-1])
                    if last_move_pos  == (x,y):
                        print('Undo Last Move')
                        if len(seq) >=3:
                            self.sgf_node.reparent(last.parent, -1)
                            n = seq[-3].properties()[-1]
                            self.last_color = C2N(n)
                        else:
                            self.sgf_node.reparent(self.sgf_node.parent, -1)
                            self.last_color = 2
                        self.state[x,y] = 2
                        return

            self.last_x = x
            self.last_y = y
            print('{}: {}-{}'.format(c_str, x+1, y+1))
            self.state = state
            self.last_color = c

            if self.GS == GameState.RUNNING:
                self.sgf_node.set_move(c_str.lower(),(x,y))
                self.sgf_node = self.sgf.extend_main_sequence()
            if KATRAIN is not None:
                KATRAIN.send([c_str.lower(), x, y])

class MotionDetection:
    def __init__(self, img):
        self.lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        img = cv2.resize(img, None, fx=0.25,fy=0.25)
        self.p0 = cv2.goodFeaturesToTrack(img, mask = None, **self.feature_params)
        self.imgLast = img
        self.hist = 0

    def hasMotion(self, img):
        img = cv2.resize(img, None, fx=0.25,fy=0.25)
        if img.shape != self.imgLast.shape:
            #first iteration after vp detect
            self.imgLast = img
            return True
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.imgLast, img, self.p0, None, **self.lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]

        motion = np.max(np.abs(good_new-good_old))
        self.p0 = good_new.reshape(-1,1,2)
        if motion >= 0.2:
            self.hist = self.hist // 2
            return True
        else: 
            if self.hist < 10:
                self.hist += 1
                return True
            else:
                return False

if __name__ == '__main__':
    parse = argparse.ArgumentParser('PyGO')
    parse.add_argument('--video', default='')
    parse.add_argument('--seek',  default=0.0, type=float)
    args = parse.parse_args()

    if args.video !='':
        webcam = cv2.VideoCapture(args.video)
        if args.seek > 0.0:
            # seek is in seconds -> mult with frames
            webcam.set(cv2.CAP_PROP_POS_MSEC, (1000*args.seek)-1) 
    else:
        raise FileNotFoundError('No Video specified!')

    ELSD = PyELSD()
    PLOT_CIRCLES=False
    PLOT_LINES=False
    global COUNT 
    COUNT = 0
    CC = CameraCalib(np.load('../config/calib_960.npy'))


    _, img = webcam.read()
    last_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    CC.center = (last_img.shape[1]//2, last_img.shape[0]//2)
    MD = MotionDetection(last_img)
    BOARD = GoBoard(CC)
    GS = Game()
    KATRAIN = None
    PatchClassifier = GoClassifier()
    #PatchClassifier = HaarClassifier()
    vps_list_1 = []
    vps_list_2 = []

    print('PyGO - Visual Interface for KaTrain')
    print('(c)alibrate      (t)rain')
    print('(n)ew game       (f)nish game')
    print('(a)nalyze')
    
    # init
    _,img_c = webcam.read()
    img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    print('Calibration')
    BOARD.calib(img)
    
    print('New Game started')
    GS.startNewGame(19)


    while True:
        #webcam.grab()
        #_,img_c = webcam.retrieve()
        _,img_c = webcam.read()
        img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

        #last_key = cv2.waitKey(1) 
        #if last_key == ord('c'):
        #    print('Calibration')
        #    BOARD.calib(img)
      
        #elif last_key == ord('t'):
        #    print('Clear Board - Press (c)ontinue')
        #    patches = []
        #    while True:
        #        ret, img_c = webcam.read()
        #        img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
        #        last_key = cv2.waitKey(1) 

        #        if not BOARD.hasEstimate:
        #            print("Calibrate the Board first!")
        #            break

        #        if last_key == ord('c'):
        #            patches = BOARD.extractOnPoints(img)
        #            PatchClassifier.train(patches)
        #            PatchClassifier.store()
        #            break
        
        #elif last_key == ord('n'):
        #elif last_key == ord('a'):
        #    print('Live Gama analysis')
        #    addr = input('Server Address: default[127.0.0.1:8888]') or 'localhost:8888'
        #    addr = addr.split(':')
        #    net_addr = (addr[0], int(addr[1]))
        #    KATRAIN = Client(net_addr, authkey=b'katrain')
        #    print('Connected to Katrain')

        #elif last_key == ord('f'):
        #    print('Game finished')
        #    GS.endGame()
        #    if KATRAIN is not None:
        #        KATRAIN.close()

        #elif last_key == ord('q'):
        #    print('Good Bye!')
        #    if GS.GS == GameState.RUNNING:
        #        GS.endGame()
        #    if KATRAIN is not None:
        #        KATRAIN.close()
        #    break
        
        if BOARD.hasEstimate:
            pdb.set_trace()
            img = BOARD.extract(img)
            if not MD.hasMotion(img):
                if PatchClassifier.hasWeights:
                    val = PatchClassifier.predict(BOARD.imgToPatches(img))
                    GS.updateState(val)

                        

        if BOARD.hasEstimate:
            img = plot_overlay(GS.state, BOARD.go_board_shifted, img)
        cv2.imshow('PyGO',img)

        last_img = img
   


#        val = analyze_board(cl2, ct2, cr2, cb2, img_c_trim)

            
# When everything done, release the capture
webcam.release()
cv2.destroyAllWindows()
