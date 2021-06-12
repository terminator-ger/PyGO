from __future__ import division 
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
from train.feature import get_feat_vec
from scipy.spatial import distance_matrix
from train.GoNet import GoNet
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
from train.feature import get_feat_vec
import torch as th
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from sgfmill import sgf

import warnings
warnings.filterwarnings('always') 
def toNP(x):
    return x.detach().cpu().numpy()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def toByteImage(image):
    '''
    Returns the uint8 representation of an image
    :param image:
    :return:
    '''
    if np.max(image) <= 1.0 or np.max(image) == 0:
        image = (image*255)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image

def toFloatImage(image):
    '''
    Returns the Float32 representation of an image
    :param image:
    :return:
    '''
    if np.max(image) > 1 or np.max(image) == 0:
        image = (image/255).astype(np.float32)
    return image

def toDoubleImage(image):
    '''
    Returns the Float32 representation of an image
    :param image:
    :return:
    '''
    if np.max(image) > 1 or np.max(image) == 0:
        image = (image/255).astype(np.double)
    return image


def toGrayImage(image):
    '''
    Reduces the image to a gray image
    Does NOT change Byte <-> Float, image is left in original datatype
    :param image:
    :return:
    '''
    if len(np.shape(image))>2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


from scipy.stats import skew, moment
from skimage.feature import hog
import numpy as np
from skimage import exposure, img_as_float
import pdb
from sklearn.linear_model import SGDClassifier
class GoClassifier:
    def __init__(self) -> None:
        self.hasWeights = False
        self.model = GoNet()
        self.load()

    def predict(self, patches):
        x = []
        for i in range(len(patches)):
            x.append(cv2.cvtColor(np.array(patches[i]).astype(np.uint8), cv2.COLOR_GRAY2RGB))
        x = th.from_numpy(np.array(x).astype(np.float32)).permute(0,3,1,2)
        lbl = self.model(x)
        lbl = lbl.detach().cpu().numpy()
        lbl = np.argmax(lbl, axis=1)
        return lbl



    def train(self, patches):
        x_train = []
        y_train = []
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 20% of all images
            iaa.imgcorruptlike.Brightness((1,3)),
            iaa.color.MultiplyAndAddToBrightness(),
        ])
        patches_arr = [[],[],[]]
        for c in range(len(patches)):
            for i in range(len(patches[c])):
                patches_arr[c].append(cv2.cvtColor(np.array(patches[c][i]).astype(np.uint8), cv2.COLOR_GRAY2RGB))

        for i in range(15):
            patches_mod = [[],[],[]]
            for c in range(0,2):
                patches_mod[c] = seq(images=patches_arr[c])
                for p in patches_mod[c]:
                    p = toDoubleImage(np.array(p))
                    x_train.append(p)
                    y_train.append(c)
        c = 2
        patches_mod[c] = seq(images=patches_arr[c])
        for p in patches_mod[c]:
            p = toDoubleImage(np.array(p))
            x_train.append(p)
            y_train.append(c)
            # reduce

   
        X = np.array(x_train)
        y = np.array(y_train)
        idx = np.arange(len(X))
        np.random.shuffle(idx)

        # shuffle
        X = X[idx]
        y = y[idx]

        SPLT = 100
        X_train = X[:-SPLT]
        y_train = y[:-SPLT]
        X_test  = X[-SPLT:]
        y_test  = y[-SPLT:]
        
        X_train = th.from_numpy(X_train.astype(np.float32))
        y_train = th.from_numpy(y_train.astype(np.int_))
        X_test  = th.from_numpy(X_test.astype(np.float32))
        y_test  = th.from_numpy(y_test.astype(np.int_))

        # channels to pos 1
        X_train = X_train.permute(0,3,1,2)
        X_test  = X_test.permute(0,3,1,2)

        batch_size = X_train.size()[0]
        opt = th.optim.Adam(self.model.parameters(), lr=0.0005)
        loss_fn = th.nn.CrossEntropyLoss()
        print('train')
        for i in range(10):
            #permutation = th.randperm(X_train.size()[0])
            permutation = np.arange(X_train.size()[0])
            
            for j in range(0, X_train.size()[0], batch_size):
                indices = permutation[j:j+batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
                y_pred = self.model(batch_x)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                opt.step()
                opt.zero_grad()

            with th.no_grad():
                y_pred = self.model(X_test)
                loss_test = loss_fn(y_pred, y_test)
                y_pred = F.log_softmax(y_pred, -1)
                y_pred = toNP(y_pred)
                y_pred = np.argmax(y_pred, axis=1)
                f1 = f1_score(y_test, y_pred, average='micro')
                print("Epoch {} : {:0.2f}, {:0.3f}, {:0.3f}".format(i,f1, toNP(loss), toNP(loss_test)))

            if i % 5 == 0:
                th.save(self.model, 'weights_{}_{}.pt'.format(i, f1))
                plt.subplot(131)
                if np.any((y_pred==0)):
                    plt.imshow(np.vstack(X_test[y_pred==0, 0]))
                plt.subplot(132)
                if np.any((y_pred==1)):
                    plt.imshow(np.vstack(X_test[y_pred==1, 0]))
                plt.subplot(133)
                if np.any((y_pred==2)):
                    plt.imshow(np.vstack(X_test[y_pred==2, 0]))
                plt.savefig('{}.png'.format(i), dpi=400)

        print(classification_report(y_test, y_pred))
        self.hasWeights = True

    def load(self):
        if os.path.exists('weights.pt'):
            self.model = th.load('weights.pt')
            self.hasWeights = True
        else:
            print('Failed to Restore Classification Alg')
            self.hasWeights = False

    def store(self):
        th.save(self.model, 'weights.pt')


class KNNClassifier:
    def __init__(self):
        self.hasWeights = False
        self.model = KNeighborsClassifier()
        #self.model = SGDClassifier(loss='hinge')
        #self.load()

    def get_feat_vec(self, d):
        feat = []
        assert(d.dtype ==  np.float64)
        assert(d.shape == (32,32))
        assert(d.max() <= 1.0)

        fd, hog_image = hog(d, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True)
        hi = exposure.rescale_intensity(hog_image, in_range=(0, 10))
       # feat.append(np.mean(d.reshape(-1)))
       # feat.append(np.var(d.reshape(-1)))
       # feat.append(skew(d.reshape(-1)))
       # feat.append(skew(d.reshape(-1)))
       # feat.append(moment(d.reshape(-1)))
        feat.append(hi.reshape(-1))
        n_win = 4
        w_size = 32 //n_win
        hist = []
        for i in range(0, 32, w_size):
            for j in range(0, 32, w_size):
                hist.append(np.histogram(d[i:i+w_size,j:j+w_size], 8, (0,1))[0])
        hist = np.array(hist)
        hist = np.sum(hist, axis=0)
        hist = hist/hist.max()
        feat.append(hist)

        return np.hstack(feat)

    def predict(self, patches):
        x_train = []
        for p in patches:
            p = toDoubleImage(p)
            feat = self.get_feat_vec(p)
            x_train.append(feat)
        X = np.array(x_train)
        lbl = self.model.predict(X)
        return lbl

    def train(self, patches):
        x_train = []
        y_train = []
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            iaa.imgcorruptlike.Brightness((1,1)),
            #iaa.color.MultiplyAndAddToBrightness(),
        ])
        patches_arr = [[],[],[]]
        for c in range(len(patches)):
            for i in range(len(patches[c])):
                patches_arr[c].append(np.array(patches[c][i]*255).astype(np.uint8))

        for i in range(5):
            patches_mod = [[],[],[]]
            for c in range(0,2):
                patches_mod[c] = seq(images=patches_arr[c])
                for p in patches_mod[c]:
                    p = toDoubleImage(np.array(p))
                    feat = self.get_feat_vec(p)
                    x_train.append(feat)
                    y_train.append(c)
        c = 2
        patches_mod[c] = seq(images=patches_arr[c])
        for p in patches_mod[c]:
            p = toDoubleImage(np.array(p))
            feat = self.get_feat_vec(p)
            x_train.append(feat)
            y_train.append(c)
            # reduce

   
        X = np.array(x_train)
        y = np.array(y_train)
        self.model.fit(X,y)
        self.hasWeights = True

    def load(self):
        if os.path.exists('knnc.joblib'):
            self.model = load('knnc.joblib')
            self.hasWeights = True
        else:
            print('Failed to Restore Classification Alg')
            self.hasWeights = False

    def store(self):
        dump(self.model, 'knnc.joblib') 




class BOWClassifier:
    def __init__(self):
        self.BOW = []
        self.model = KNeighborsClassifier()
        self.extractor = cv2.SIFT_create()
        self.km = KMeans(3)
        self.hasWeights = False
        self.load()

    def predict(self, patches):
        x_train = []
        for p in patches:
            p = toByteImage(p)
            _, feat = self.extractor.detectAndCompute(p, None)
            if feat is not None:
                h = self.descToHist(feat)
                x_train.append(h)
        X = np.array(x_train)
        lbl = self.model.predict(X)
        return lbl

    def train(self, patches):
        label = []
        self.BOW = []
        for c, patch in enumerate(patches):
            c_train = []
            for p in patch:
                p = toByteImage(np.array(p))
                _, feat = self.extractor.detectAndCompute(p, None)
                if feat is not None:
                    c_train.append(feat)
                    label.append(c)
            # reduce
            self.km.fit(np.vstack(c_train))
            self.BOW.append(self.km.cluster_centers_)

        self.BOW = np.array(self.BOW).reshape(-1,128)
        x_train = []
        y_train = []
        for c, patch in enumerate(patches):
            for p in patch:
                p = toByteImage(np.array(p))
                _, feat = self.extractor.detectAndCompute(p, None)
                if feat is not None:
                    h = self.descToHist(feat)
                    x_train.append(h)
                    y_train.append(c)
    
        X = np.array(x_train)
        y = np.array(y_train)
        self.model.fit(X,y)
        self.hasWeights = True

    def descToHist(self, desc):
        D = distance_matrix(desc, self.BOW)
        nn = np.argmin(D, axis=1)
        id, cnt = np.unique(nn, return_counts=True)
        hist = np.zeros(len(self.BOW))
        hist[id] = cnt
        return hist

    def load(self):
        if os.path.exists('knn.joblib') and os.path.exists('bow.joblib'):
            self.model = load('knn.joblib')
            self.BOW = load('bow.joblib')
            self.hasWeights = True
        else:
            print('Failed to Restore Classification Alg')
            self.hasWeights = False

    def store(self):
        dump(self.model, 'knn.joblib') 
        dump(self.BOW, 'bow.joblib') 

import sys
sys.path.insert(0, './train')

def toNP(x):
    return x.detach().cpu().numpy()

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

def clear(lines):
    # cluster x
    from scipy.spatial.distance import pdist
    from scipy.spatial import  distance_matrix
    from sklearn.cluster import KMeans, AgglomerativeClustering
    circ_dists = distance_matrix(lines, lines, 2)
    x_data = lines[:,0]
    # sort by closes two dists
    N = 3
    circ_dists = np.sort(circ_dists, axis=0)
    circ_dists_idx = np.argsort(circ_dists, axis=0)
    circ_dists = circ_dists[1:(1+N)]
    med_dist = np.median(circ_dists)
    circ_dists = np.square(circ_dists-med_dist)

    circ_dists_idx = circ_dists_idx[1:(1+N)]
    km = AgglomerativeClustering(n_clusters=None, 
                                linkage='single',
                                distance_threshold=5)
    lblx = km.fit_predict(circ_dists.T)
    #plt.scatter(lines[:,0], lines[:,1], c=lblx)
    #plt.show()
    _, xcount = np.unique(lblx, return_counts=True)
    target_cluster = np.argmax(xcount)
    delx = np.argwhere(lblx != target_cluster)
    lines = np.delete(lines, delx, axis=0)
    lblx = np.delete(lblx, delx)

    #plt.scatter(lines[:,0], lines[:,1], c=lblx)
    #plt.show()
    #pdb.set_trace()
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
    from scipy.cluster.hierarchy import linkage, average, fcluster
    arr = np.concatenate((x,y)).reshape(-1,2)
    X = average(arr.reshape(-1,2))
    cl = fcluster(X, 10, criterion='distance')
    #c = []
    #for i in range(1, cl.max()):
    #c.append(np.mean(arr[np.argwhere(cl==i),:], axis=0))
    c = [np.mean(arr[np.argwhere(cl==i),:], axis=0) for i in range(1,cl.max())]
    c = np.array(c).reshape(-1,2)
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

        lines_raw_orig = cluster(lines[0], lines[1])
        lines_raw = cluster(lines_warpedH[0], lines_warpedV[0])
        lines = clear(lines_raw)

        img_max = (img_cw.shape[1], img_cw.shape[0])
        img_min = (0,0)
    
        self.grid = get_ref_go_board_coords(np.min(lines, axis=0), np.max(lines, axis=0))
        img_cw = cv2.warpPerspective(img, H, img_limits)

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

        reg = AffineRegistration(**{'X': lines, 
                                'Y': self.grid, 
                                'max_iterations': 8000,
                                'tolerance': 0.0000001,
                                })
        # ty, param = reg.register(callback)
        ty, param = reg.register()

        R = np.eye(3)
        R[0:2,0:2]=param[0]
        R[0:2,2]=param[1]


        w_board = cv2.transform(np.array([self.grid]), (R))[0][:,:2]
        ow_board = cv2.perspectiveTransform(np.array([self.grid]), R@np.linalg.inv(H))[0]

        #img_grid = plot_grid(img.copy(), ow_board.reshape(-1,2))
        #cv2.imshow('PyGO', img_grid)
        #cv2.waitKey(500)        
    
        src_pt = find_src_pt(ow_board, lines_raw_orig)
        H_refined = cv2.findHomography(src_pt, ow_board)[0]
        w_board = cv2.perspectiveTransform(np.array([self.grid]), R @ np.linalg.inv(H) @ np.linalg.inv(H_refined))[0]

        H_refined = R @ np.linalg.inv(H) @ np.linalg.inv(H_refined)

        img_grid = plot_grid(img.copy(), w_board.reshape(-1,2))
        cv2.imshow('PyGO', img_grid)
        cv2.waitKey(1500)       
        self.H = H_refined 
        self.hasEstimate=True

        print(self.H)
 
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

class Color(Enum):
    WHITE = 0
    BLACK = 1
    NONE  = 2

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
        if np.count_nonzero(np.abs(self.state-state)) > 1:
            return
        # get last moves color
        idx = np.argwhere(np.abs(self.state-state)>0)
        if idx.size==0:
            return
        x = idx[0,0]
        y = idx[0,1]
        c = state[x,y]
        if c != self.last_color:
            c_str = 'w' if c == 0 else 'b'
            self.last_x = x
            self.last_y = y
            print('{}: {}-{}'.format(c_str, x+1, y+1))
            self.state = state
            self.last_color = c

            if self.GS == GameState.RUNNING:
                self.sgf_node.set_move(c_str,(x,y))
                self.sgf_node = self.sgf.extend_main_sequence()
            if KATRAIN is not None:
                KATRAIN.send([c_str, x, y])

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
    ELSD = PyELSD()
    PLOT_CIRCLES=False
    PLOT_LINES=False
    global COUNT 
    COUNT = 0
    CC = CameraCalib(np.load('../calib_960.npy'))

    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _, img = webcam.read()
    last_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    MD = MotionDetection(last_img)
    BOARD = GoBoard(CC)
    GS = Game()
    KATRAIN = None
    #PatchClassifier = BOWClassifier()
    #PatchClassifier = KNNClassifier()
    PatchClassifier = GoClassifier()
    vps_list_1 = []
    vps_list_2 = []

    print('PyGO - Visual Interface for KaTrain')
    print('(c)alibrate      (t)rain')
    print('(n)ew game       (f)nish game')
    print('(a)nalyze')

    while True:
        #ret, img_c = webcam.read()
        webcam.grab()
        _,img_c = webcam.retrieve()
        img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

        last_key = cv2.waitKey(1) 
        if last_key == ord('c'):
            print('Calibration')
            BOARD.calib(img)
      
        elif last_key == ord('t'):
            print('Clear Board - Press (c)ontinue')
            patches = []
            while True:
                ret, img_c = webcam.read()
                img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
                last_key = cv2.waitKey(1) 

                if not BOARD.hasEstimate:
                    print("Calibrate the Board first!")
                    break

                if last_key == ord('c'):
                    patches = BOARD.extractOnPoints(img)
                    PatchClassifier.train(patches)
                    PatchClassifier.store()
                    break
        
        elif last_key == ord('n'):
            print('New Game started')
            GS.startNewGame(19)

        elif last_key == ord('a'):
            print('Live Gama analysis')
            addr = input('Server Address: default[127.0.0.1:8888]') or 'localhost:8888'
            addr = addr.split(':')
            net_addr = (addr[0], int(addr[1]))
            KATRAIN = Client(net_addr, authkey=b'katrain')
            print('Connected to Katrain')

        elif last_key == ord('f'):
            print('Game finished')
            GS.endGame()
            if KATRAIN is not None:
                KATRAIN.close()

        elif last_key == ord('q'):
            print('Good Bye!')
            if GS.GS == GameState.RUNNING:
                GS.endGame()
            if KATRAIN is not None:
                KATRAIN.close()
            break
        
        if BOARD.hasEstimate:
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
