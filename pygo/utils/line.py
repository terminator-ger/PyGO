import numpy as np
import pdb
import cv2
from inpoly import inpoly2
from pygo.utils.typing import Lines, Line

def warp_lines(lines, H):
    ones = np.ones((lines.shape[0], 1))
    l_homo = np.concatenate((lines, ones), axis=1)
    l_warped = l_homo @ H
    return l_warped


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

def warp_lines(lines, H):
    l = len(lines)
    H = np.tile(H, (l,1,1))
    src_pt_H = cv2.convertPointsToHomogeneous(lines.reshape(1,-1,2))
    warped_pt = H @ src_pt_H.transpose(0,2,1)
    dst_pt = np.squeeze(cv2.convertPointsFromHomogeneous(warped_pt))
    return dst_pt
        

def contour_to_lines(contour):
    lines = []
    for idx in range(len(contour)):
        lines.append(np.vstack((contour[idx], contour[(idx+1)%len(contour)])))
    return np.vstack(lines)

def is_line_within_board(l, board_lines):
    board_lines = contour_to_lines(board_lines)
    for bline in board_lines:   
        print(bline)
        print(l)
        ret = intersection(line(l[0], l[1]), line(bline[0],bline[1]))
        # check for intersection point on pol
        IN, ON = inpoly2(np.array([ret]), board_lines)
        pdb.set_trace()
        if not np.all(np.logical_or(IN, ON)):
            return False
        else:
            r = inpoly2(l, board_lines)
            pdb.set_trace()
            #check for intersection point within contour
        print(ret)

    return True

def points_in_polygon(polygon, pts):
    '''
    https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    by Ta946
    '''
    pts = np.asarray(pts,dtype='float32')
    polygon = np.asarray(polygon,dtype='float32')
    contour2 = np.vstack((polygon[1:], polygon[:1]))
    test_diff = contour2-polygon
    mask1 = (pts[:,None] == polygon).all(-1).any(-1)
    m1 = (polygon[:,1] > pts[:,None,1]) != (contour2[:,1] > pts[:,None,1])
    slope = ((pts[:,None,0]-polygon[:,0])*test_diff[:,1])-(test_diff[:,0]*(pts[:,None,1]-polygon[:,1]))
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)
    m3 = (slope < 0) != (contour2[:,1] < polygon[:,1])
    m4 = m1 & m3
    count = np.count_nonzero(m4,axis=-1)
    mask3 = ~(count%2==0)
    mask = mask1 | mask2 | mask3
    return mask
     

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


def min_line_endpoint_dist(l1: Line, l2: Line) -> float:
    dx1 = l1[0] - l2[0]
    dy1 = l1[1] - l2[1]

    dx2 = l1[0] - l2[2]
    dy2 = l1[1] - l2[3]

    dx3 = l1[2] - l2[0]
    dy3 = l1[3] - l2[1]

    dx4 = l1[2] - l2[2]
    dy4 = l1[3] - l2[3]

    d1 = np.sqrt(dx1*dx1 + dy1*dy1)
    d2 = np.sqrt(dx2*dx2 + dy2*dy2)
    d3 = np.sqrt(dx3*dx3 + dy3*dy3)
    d4 = np.sqrt(dx4*dx4 + dy4*dy4)
    return min(d1,d2,d3,d4)

def line_angle_diff(l1: Line, l2: Line) -> float:
    dx1 = l1[0] - l1[2]
    dy1 = l1[1] - l1[3]
    a1 = np.arctan2(dy1, dx1)
    dx1 = l1[2] - l1[0]
    dy1 = l1[3] - l1[1]
    a11 = np.arctan2(dy1, dx1)


    dx2 = l2[0] - l2[2]
    dy2 = l2[1] - l2[3]
    a2 = np.arctan2(dy2, dx2)
    dx2 = l2[2] - l2[0]
    dy2 = l2[3] - l2[1]
    a22 = np.arctan2(dy2, dx2)
    a1 = max(a1, a11)
    a2 = max(a2, a22)

    #print("A1: {}".format(a1/np.pi*180))
    #print("A2: {}".format(a2/np.pi*180))

    return np.abs(a1 - a2)

def merge_lines(l1: Line, l2: Line) -> Line:
    dx1 = l1[0] - l2[0]
    dx2 = l1[0] - l2[2]
    dx3 = l1[2] - l2[0]
    dx4 = l1[2] - l2[2]

    dy1 = l1[1] - l2[1]
    dy2 = l1[1] - l2[3]
    dy3 = l1[3] - l2[1]
    dy4 = l1[3] - l2[3]

    d1 = np.sqrt(dx1*dx1 + dy1*dy1)
    d2 = np.sqrt(dx2*dx2 + dy2*dy2)
    d3 = np.sqrt(dx3*dx3 + dy3*dy3)
    d4 = np.sqrt(dx4*dx4 + dy4*dy4)   

    max_dist_idx = np.argmax([d1,d2,d3,d4])
    if max_dist_idx == 0:
        return np.array([l1[0],l1[1], l2[0],l2[1]])
    elif max_dist_idx == 1:
        return np.array([l1[0],l1[1], l2[2],l2[3]])
    elif max_dist_idx == 2:
        return np.array([l1[2],l1[3], l2[0],l2[1]])
    elif max_dist_idx == 3:
        return np.array([l1[2],l1[3], l2[2],l2[3]])


def merge_lines_mean(l1:Line, l2:Line) -> Line:
    dx1 = l1[0] - l2[0]
    dy1 = l1[1] - l2[1]

    dx2 = l1[0] - l2[2]
    dy2 = l1[1] - l2[3]

    d1 = np.sqrt(dx1*dx1 + dy1*dy1)
    d2 = np.sqrt(dx2*dx2 + dy2*dy2)

    max_dist_idx = np.argmin([d1,d2])
    if max_dist_idx == 0:
        return np.array([(l1[0]+l2[0])/2, (l1[1]+l2[1])/2, 
                         (l1[2]+l2[2])/2, (l1[3]+l2[3])/2])
    elif max_dist_idx == 1:
        return np.array([(l1[0]+l2[2])/2, (l1[1]+l2[3])/2, 
                         (l1[2]+l2[0])/2, (l1[3]+l2[1])/2])


def point_in_circle(points, circle):
    if len(points.shape) == 1:
        #only one point
        points = points.reshape(-1,2)
    if len(circle.shape) == 1:
        circle = circle.reshape(-1,3)

    dist =  (points[:,0] - circle[:,0])**2 + (points[:,1] - circle[:,1])** 2 
    return np.less(dist, circle[:,2]**2)