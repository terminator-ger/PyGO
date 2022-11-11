import cv2
import numpy as np

from typing import List, Tuple

def coordinate_to_letter(x: int) -> str:
    table = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 
             'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
    return table[x]

def pygo_to_go_coord_sys(c: Tuple[int, int], board_size=19) -> Tuple[str, int]:
    x, y = c
    return (coordinate_to_letter(x), board_size-y) 

def sgfmill_to_pygo_coord_sys(c: Tuple[int, int], board_size=19) -> Tuple[int, int]:
    x,y = c
    return (y, board_size - x)


def get_ref_go_board_coords(min, max, border):
    # assume symmectric go board
    dpx = (max[0]-min[0]) / 19
    dpy = (max[1]-min[1]) / 19
    go_board = np.zeros((19, 19, 2))
    for i in range(19):
        for j in range(19):
            go_board[i, j, 0] = i * dpx 
            go_board[i, j, 1] = j * dpy 
    
    go_board += np.array([border, border])
    return go_board.reshape(-1,2)

def get_ref_coords(shape, border):
    if len(shape) == 3:
        h,w,_ = shape
    elif len(shape) == 2:
        h,w = shape
    side = min(h,w)
    dpx = (side-2*border) / 19
    dpy = (side-2*border) / 19
    go_board = np.zeros((19, 19, 2))
    for i in range(19):
        for j in range(19):
            go_board[i, j, 0] = i * dpx 
            go_board[i, j, 1] = j * dpy 
    go_board += np.array([border, border])
    return go_board.reshape(-1,2)

def get_grid_lines(grid):
    grid = grid.reshape(19,19,2) 
    x_min = grid[0,:,0].min()
    x_max = grid[18,:,0].min()
    y_min = grid[:,0,1].max()
    y_max = grid[:,18,1].max()
    dx = x_max - x_min
    dy = y_max - y_min
    ddx = dx / 18
    ddy = dy / 18
    border = 50
    bhalf = border/2
    grid_img = np.ones((int(dy)+border,int(dx)+border,3),dtype=np.uint8)*255

    lines = np.zeros((19*2,4))

    for i in range(19):
        lines[i] = [x_min, i*ddy, x_max, i*ddy]
        pt1 = np.array([bhalf, bhalf+i*ddy], dtype=int)
        pt2 = np.array([dx+bhalf,bhalf+i*ddy], dtype=int)
        cv2.line(grid_img, pt1, pt2, (0,0,255), 1)
        
        lines[19+i] = [i*ddx, y_min, i*ddx, y_max]
        pt1 = np.array([bhalf+i*ddx, bhalf], dtype=int)
        pt2 = np.array([bhalf+i*ddx, dy+bhalf], dtype=int)
        cv2.line(grid_img, pt1, pt2, (0,0,255), 1)
    
    for pt1 in [int(3*ddx+bhalf), int(9*ddx+bhalf), int(15*ddx+bhalf)]:
        for pt2 in [int(3*ddy+bhalf), int(9*ddy+bhalf), int(15*ddy+bhalf)]:
            cv2.circle(grid_img, (pt1, pt2), color=(0,0,255), radius=3, thickness=-1)
    updated_grid = np.zeros((19,19,2))
    for i,x in enumerate(np.linspace(bhalf, bhalf+dx, 19)):
        for j,y in enumerate(np.linspace(bhalf, bhalf+dy, 19)):
            updated_grid[i,j] = [x,y]

    return lines, grid_img, updated_grid


def flattenList(l: List) -> List:
    return [item for sublist in l for item in sublist]

def toNP(x):
    return x.detach().cpu().numpy()


def clear(lines):

    from scipy.spatial import  distance_matrix
    from sklearn.cluster import AgglomerativeClustering


    # cluster x
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
                                distance_threshold=4)
    lblx = km.fit_predict(circ_dists.T)
    #plt.scatter(lines[:,0], lines[:,1], c=lblx)
    #plt.show()
    _, xcount = np.unique(lblx, return_counts=True)
    target_cluster = np.argmax(xcount)
    delx = np.argwhere(lblx != target_cluster)
    lines = np.delete(lines, delx, axis=0)
    lblx = np.delete(lblx, delx)


    return lines


def cluster(x,y):
    from scipy.cluster.hierarchy import average, fcluster
    arr = np.concatenate((x,y)).reshape(-1,2)
    X = average(arr.reshape(-1,2))
    cl = fcluster(X, 10, criterion='distance')
    #c = []
    #for i in range(1, cl.max()):
    #c.append(np.mean(arr[np.argwhere(cl==i),:], axis=0))
    c = [np.mean(arr[np.argwhere(cl==i),:], axis=0) for i in range(1,cl.max())]
    c = np.array(c).reshape(-1,2)
    return c

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


def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def mask_board(image, go_board, border_size=15):
    mask = np.zeros_like(image)
    go = go_board.reshape(19,19,2)
    corners = np.array([go[0,0], 
                        go[0,18], 
                        go[18,18], 
                        go[18,0]]).reshape(-1,1,2).astype(int)
    mask = cv2.fillConvexPoly(mask, corners, (255,255,255))
    mask = cv2.dilate(mask, np.ones((3,3)), iterations=border_size)
    if len(mask.shape) == 3:
        rect = cv2.boundingRect(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))               # function that computes the rectangle of interest
    else:
        rect = cv2.boundingRect(mask.astype(np.uint8))
    x,y,w,h = rect
    cropped_img = image[y:y+h, x:x+w]#.copy()
    return cropped_img, (x,y)

def find_src_pt(go, lines):
    # find nn
    circ_dists = distance_matrix(go, lines, 2)
    nn_idx = np.argmin(circ_dists, axis=1)
    nn = lines[nn_idx,:]
    # remove unmatched
    return nn


def cv2Input() -> str:
    ''' blocks until enter is pressed
    '''
    _ipt = 0
    _str = []
    while (_ipt != 13):
        _ipt = cv2.waitKey(0)
        _str.append(chr(_ipt%256) if _ipt > 47 else "")
    _str = "".join(_str)
    return _str

