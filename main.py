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
from pycpd import AffineRegistration
from functools import partial
from scipy import ndimage
from skimage.exposure import equalize_hist
from joblib import load, dump
from sklearn.mixture import GaussianMixture

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
from train.GoNet import GoNet
import torch as th
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

def analyze_board(l, r, t, b, img):
    i = 0
    global COUNT
    val_lbl = np.zeros((19,19))
    patches = []
    for path in zip(l,r,t,b):
        l1 = np.array([path[0][0], path[1][1]])
        l2 = np.array([path[2][0], path[1][1]])
        l3 = np.array([path[2][0], path[3][1]])
        l4 = np.array([path[0][0], path[3][1]])
        p = np.array([l1,l2,l3,l4]).astype(int)
        patch = cv2.cvtColor(crop(p, img), cv2.COLOR_BGR2GRAY)
        #patch = crop(p, img)
        patch = patch / 255.0

        patch =  transform.resize(patch, (32,32),  anti_aliasing=True)
        patches.append(patch)
        #m = np.nanmean(patch)

    cross = np.zeros((9,9), dtype=np.float32)*-255.0
    cross[:,4] = 255.0
    cross[4,:] = 255.0
    circle = np.zeros((13,13), dtype=np.float32)
    circle = cv2.circle(circle, (6,6), 6, 255.0, 1)

    val = []
#    for i, patch in enumerate(patches):
#        patch =  transform.resize(patch, (32,32),  anti_aliasing=True)
#        #cv2.imwrite('train/3/{}.png'.format(COUNT), patch*255)
#        COUNT += 1
#        if len(patch.shape)==3:
#            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
#        patch = patch.astype(float)
#        patch /= 255.0
#        val.append(get_feat_vec(patch))
#
#    val = np.array(val)
#
#    val_lbl = SVM.predict(val)
    X = th.from_numpy(np.stack(patches)).permute(0,3,1,2)
    pdb.set_trace()
    val_lbl = GoNet(X)
    val_lbl = toNP(val_lbl)
    val_lbl = np.argmax(val_lbl, axis=1)
        
    return val_lbl

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
    for v, c in zip(val, coords): 
        #if v > 0.6:
        if v == 1:
            #black
            color = (255, 255, 255)
            thickness=-1
        #elif v < 0.4:
        elif v == 2:
            #white
            color = (0,0,0)
            thickness=-1
        elif v == 0:
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
    pdb.set_trace()

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
    rect = cv2.boundingRect(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))               # function that computes the rectangle of interest
    x,y,w,h = rect
    cropped_img = image[y:y+h, x:x+w].copy()
    return cropped_img, (x,y)

if __name__ == '__main__':
    ELSD = PyELSD()
    PLOT_CIRCLES=False
    PLOT_LINES=False
    global COUNT 
    COUNT = 0
    intr = np.load('../calib.npy')
    dist = np.load('../dist.npy')
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cx = intr[0,2]
    cy = intr[1,2]
    #gmm = load('gmm.joblib')
    #SVM = load('train/svm.joblib')
    GoNet = th.load('train/weights_gonet.pt')

    print("Estimating Homography")
    vps_list_1 = []
    vps_list_2 = []

    print('Align Camera - press (c)ontinue')
    img_buf = collections.deque(maxlen=1)
    while True:
        ret, img_c = webcam.read()
        img_buf.append(img_c)
        cv2.imshow('PyGO',img_c)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
          
    #for i in tqdm(range(30)):
        #ret, img_c = webcam.read()
    
    vp = VPDetection(focal_length=intr[0,0], principal_point=(cx,cy), length_thresh=20)
    lines_vp = []
    for img_c in tqdm(img_buf):
        #/cv2.imwrite('white.png',img_c)
        # gather initial vps to determine stable vps
        if ret:
            if len(img_c.shape) == 3:
                img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
            h,w = img.shape
            van_points = vp.find_vps(img)
            #vpd = vp.create_debug_VP_image()

            vps = vp.vps_2D
            vp3d = vp.vps
            vert_vp = np.argmax(np.abs(vp3d[:,2]))
            vps = np.delete(vps, vert_vp, axis=0)
            vp1 = np.array([vps[0,0], vps[0,1], 1])
            vp2 = np.array([vps[1,0], vps[1,1], 1])
            vps_list_1.append(vp1)
            vps_list_2.append(vp2)        
            lines_cluster = vp.get_lines()
            # drop lines with least 
            del lines_cluster[(np.argmin([len(x) for x in lines_cluster]))]
            lines_vp.append(lines_cluster)

    vp1 = np.median(np.array(vps_list_1), axis=0)
    vp2 = np.median(np.array(vps_list_2), axis=0)
    warped_img, H, img_limits = compute_homography_and_warp(img, vp1, vp2, clip=False, clip_factor=1)
    img_min = (img.shape[1],img.shape[0])
    img_max = (0,0)


    img_cw = cv2.warpPerspective(img, H, img_limits)
    img_dcw = np.asarray(img_cw, dtype=np.double, order='C')


    lines_warpedV = []
    lines_warpedH = []

    for lines in lines_vp:
        lines_warpedV.append(cv2.perspectiveTransform(np.stack(lines[0]).reshape(-1,1,2), H).reshape(-1,4))
        lines_warpedH.append(cv2.perspectiveTransform(np.stack(lines[1]).reshape(-1,1,2), H).reshape(-1,4))

    i = intersect(lines_warpedH[0], lines_warpedV[0])
    img_min = min(img_min, tuple(np.squeeze(np.min(i, axis=0))))
    img_max = max(img_max, tuple(np.squeeze(np.max(i, axis=0))))
   
    go_board = get_ref_go_board_coords(img_min, img_max)
   # plt.imshow(plot_circles(img_dcw, go_board[:,0], go_board[:,1]))
   # plt.show()

    go_circle_left  = go_board - [14,0]
    go_circle_top   = go_board + [0,14]
    go_circle_right = go_board + [14,0]
    go_circle_bot   = go_board - [0,14]
    # Detect Board
    

    (e_cx, e_cy, l_s, l_e) = ELSD.detect(img_dcw)

    l_s, l_e = lines_to_2d(l_s, l_e)

    lines = l_s
    lines = np.concatenate((lines, np.array([e_cx,e_cy]).T))
    lines_original = cv2.perspectiveTransform(np.array([lines]), np.linalg.inv(H))[0]
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


    reg = AffineRegistration(**{'Y': lines, 'X': go_board, 'max_iterations': 800})
    #ty, param = reg.register(callback)
    ty, param = reg.register()

    R = np.eye(3)
    R[0:2,0:2]=param[0]
    R[0:2,2]=param[1]

    def _tform(coords, R, H):
        c = cv2.transform(np.array([coords]), np.linalg.inv(R))[0]
        c = cv2.perspectiveTransform(np.array([c[:,:2]]), np.linalg.inv(H))[0]
        return c
    w_board = cv2.transform(np.array([go_board]), np.linalg.inv(R))[0][:,:2]
    ow_board = cv2.perspectiveTransform(np.array([w_board]), np.linalg.inv(H))[0]
    
    img_grid = plot_grid(img_c, ow_board)
    cv2.imshow('PyGO', img_grid)
    cv2.waitKey(100)


#    cl = _tform(go_circle_left, R, H)
#    cr = _tform(go_circle_right, R, H)
#    ct = _tform(go_circle_top, R, H)
#    cb = _tform(go_circle_bot, R, H)

    while(True):
        ret, img_c = webcam.read()

        img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
        img_float = img/255.0
        if ret:
  #          cl2 = _tform(go_circle_left,  R, np.eye(3))
  #          cr2 = _tform(go_circle_right, R, np.eye(3))
  #          ct2 = _tform(go_circle_top,   R, np.eye(3))
  #          cb2 = _tform(go_circle_bot,   R, np.eye(3))
            w_board = cv2.transform(np.array([go_board]), np.linalg.inv(R))[0][:,:2]
            ow_board = cv2.perspectiveTransform(np.array([w_board]), np.linalg.inv(H))[0]

            img_cw = cv2.warpPerspective(img, H, img_limits)
            img_w = cv2.warpPerspective(img_c, H, img_limits)
            img_dcw = np.asarray(img_cw, dtype=np.double, order='C')


            if PLOT_CIRCLES or PLOT_LINES:
                (e_cx, e_cy, l_s, l_e) = ELSD.detect(img_dcw)
                l_s, l_e = lines_to_2d(l_s, l_e)
            if PLOT_CIRCLES:
                img_cw = plot_circles(img_cw, e_cx, e_cy)
                cv2.imshow('PyGO', img_cw)
                cv2.waitKey(1)
            if PLOT_LINES:
                img_cw = plot_lines(img_cw, l_s, l_e)

            #warped_lines = warp_lines(np.concatenate((l_s, l_e), axis=0), H)
            # warp lines back to original image
            #lines = np.concatenate((l_s, l_e), axis=0)
            #img = plot_circles(img, ow_board[:,0], ow_board[:,1])

            #img_float = normalize_board_brigthnes(ow_board, img_float)

            #img_float = equalize_hist(img_float)
            #val = analyze_board(cl, ct, cr, cb, img_float)
            #img = plot_val(val, ow_board, img)

#            val = analyze_board(cl2, ct2, cr2, cb2, img_cw)
#            img = plot_overlay(val, ow_board, img_c)


            img_c_trim, (x,y) = mask_board(img_w, w_board)
            w_board -= np.array([x,y])

            #determined spaces from grid spacing
            cell_w = np.mean(np.diff(w_board.reshape(19,19,2)[:,:,0], axis=0))
            cell_h = np.mean(np.diff(w_board.reshape(19,19,2)[:,:,1], axis=1))
            
            cl2 = w_board - np.array([cell_w/2, 0])
            cr2 = w_board + np.array([cell_w/2, 0])
            ct2 = w_board + np.array([0, cell_h/2])
            cb2 = w_board - np.array([0, cell_h/2])
 
            val = analyze_board(cl2, ct2, cr2, cb2, img_c_trim)
            img = plot_overlay(val, w_board, img_c_trim)



            cv2.imshow('PyGO', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
          
  
            
# When everything done, release the capture
webcam.release()
cv2.destroyAllWindows()