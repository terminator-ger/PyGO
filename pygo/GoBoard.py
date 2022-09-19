import numpy as np
import cv2
from inpoly import inpoly2
from lu_vp_detect import VPDetection
from skimage import data, color, img_as_ubyte, exposure, transform, img_as_float
from utils.homography import compute_homography_and_warp
import matplotlib.pyplot as plt
import math
from utils.line import intersect, is_line_within_board, warp_lines
from utils.misc import get_ref_go_board_coords, find_src_pt, mask_board
from utils.plot import plot_grid
from utils.image import toByteImage, toCMYKImage
from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration
from functools import partial
import pdb
from shapely.geometry import Polygon, LineString

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

            patch = self.crop(p, img)

            # fail for false calib -> None
            if not np.all(np.array(patch.shape) > 0):
                return None

            patch =  transform.resize(patch, (32,32),  anti_aliasing=True)
            patches.append(patch)
        return patches

    def line_point_distance(self, point, line_s, line_e):
        '''
        '''
        nom = np.abs((line_e[0]-line_s[0])*(line_s[1]-point[1]) - (line_s[0]-point[0])*(line_e[1]-line_s[1]))
        denom = np.sqrt((line_e[0]-line_s[0])**2 + (line_e[1]-line_s[1])**2) + 1e-12
        return nom/denom

    def sort_corners(self, corners):
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

    def get_corners(self, vp1, vp2, img):
        '''
        vp1: 2d vanishing point
        vp2: 2d vanishing point
        img: thresholded image of the board
        '''

        contours, hierarchy = cv2.findContours(img, 1, 2)
        # vp1 and vp2 are the horizontal and vertical vps
        min_dist = 10000
        corners = []
        corners_mat = None

        best_image = np.zeros_like(img)
        img_out = None
        min_area = np.prod(img.shape)
        print("Min Area: {}".format(min_area))
        for c, cnt in enumerate(contours):
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
                    cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
                    #text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
                    #cv2.putText(output, text, (0, 0), cv2.FONT_HERSHEY_SIMPLEX,\
                    #    0.9, (0, 255, 0), 2)
                    ## show the approximated contour image
                    #print("[INFO] {}".format(text))
                    #cv2.imshow("PyGO", output)
                    #cv2.waitKey(50)
                    # debug output end

                    if len(approx) == 4:
                        # when we found an approximation with only four corneres we can stop
                        cnt = approx
                        break

                if len(approx) != 4:
                    # in case we looped to the end without a good result goto next shape
                    print("False corner count, we have : {}".format(len(approx)))
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
                        image = cv2.drawContours(img_out, contours, c, (0, 255, 0), 3)
                        #cv2.imshow('',image)
                        #cv2.waitKey(1000)

                        best_image = image
                        best_cnt = cnt
                        #print(area)
                        corners = [leftmost, topmost, rightmost, bottommost]
                        corners_mat = np.array(corners)
        #if corners_mat is not None:
        #    plt.imshow(best_image)
        #    plt.scatter(corners_mat[:,0], corners_mat[:,1])
        #    plt.show(block=False)

        return corners_mat
    
    def detect_board_corners(self, vp1, vp2, img_bw):

        corners = self.get_corners(vp1, vp2, img_bw)
        if corners is None:
            kernelSizes = [(3,3), (3,3), (3,3)]#(5, 5), (7, 7)]
            # loop over the kernels sizes
            for kernelSize in kernelSizes:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
                img_bw = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
                corners = self.get_corners(vp1, vp2, img_bw)
                #up to threee filter layers
                if corners is not None:
                   break

        #plt.imshow(img_bw)
        #plt.scatter(corners[:,0], corners[:,1])
        #plt.show()
 
        return corners



    def get_vp(self, img):
        van_points = self.vp.find_vps(img)
        vpd = self.vp.create_debug_VP_image()
        lines_vp = []
        vps = self.vp.vps_2D
        vp3d = self.vp.vps
        vert_vp = np.argmax(np.abs(vp3d[:,2]))
        vps = np.delete(vps, vert_vp, axis=0)
        vp1 = np.array([vps[0,0], vps[0,1], 1])
        vp2 = np.array([vps[1,0], vps[1,1], 1])
        return vp1, vp2
    
   
    def calib(self, img):
        img_c = img
        if len(img.shape) == 3:
            h,w,c = img.shape
            #instead of the grayscale variant extract the red part
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = toCMYKImage(img)[:,:,3]
            #plt.imshow(img)
            #plt.show()
        else:
            h,w = img.shape

        vp1, vp2 = self.get_vp(img) 

        thresh, img_bw = cv2.threshold(img, \
                                    0, \
                                    255, \
                                    cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #cv2.imshow('bw',img_bw)
        #cv2.waitKey(1)
        # clean img_bw


        # assumption most lines in the image are from the go board -> vp give us the plane
        # the contour which belongs to those vp is the board
        corners = self.detect_board_corners(vp1, vp2, img_bw.copy())
        if corners is None:
            print('Calib Failed')
            return
        mask = np.zeros((img_bw.shape),np.uint8)
        mask = cv2.fillConvexPoly(mask, corners.astype(int), 255)

        # rectify image
        lines_vp = []
        vps_list_1 = []
        vps_list_2 = []

        vps_list_1.append(vp1)
        vps_list_2.append(vp2)        
        lines_cluster = self.vp.get_lines()
        # drop lines with least 
        del lines_cluster[(np.argmin([len(x) for x in lines_cluster]))]
        lines_vp.append(lines_cluster)

        vp1 = np.median(np.array(vps_list_1), axis=0)
        vp2 = np.median(np.array(vps_list_2), axis=0)
        img_c = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
        #cv2.drawContours(img_c, np.array([corners.astype(int)]), -1, (255, 255, 255), -1, cv2.LINE_AA)
        img_c = cv2.polylines(img_c, np.array([corners.astype(int)]), True, (0, 255, 255), 1)
        warped_img, H, img_limits = compute_homography_and_warp(img_c.copy(), vp1, vp2, clip=False, clip_factor=1)
        #warped_img, H, img_limits = compute_homography_and_warp(img, vp1, vp2, clip=False, clip_factor=1)
        self.img_limits = img_limits

        img_c = cv2.warpPerspective(img_c, H, img_limits)
        img_cw = cv2.warpPerspective(img_bw, H, img_limits)
        img_cw2 = cv2.warpPerspective(img, H, img_limits)
        mask = cv2.warpPerspective(mask, H, img_limits)

        # warp corners
        corners_w = warp_lines(corners, H)
        
        #cv2.drawContours(mask,[best_cnt],0,255,-1)
        #cv2.drawContours(mask,[best_cnt],0,0,2)
        for i in range(7):
            mask = cv2.dilate(mask, np.ones((3,3), dtype=np.uint8))
        img_cw[mask==0] = 0
        img_dcw = np.asarray(img_cw, dtype=np.double, order='C')
        #plt.imshow(img_cw)
        #plt.show()
        thresh, img_cw = cv2.threshold(img_cw, \
                                    0, \
                                    255, \
                                    cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        img_cw[mask==0] = 0
        #plt.imshow(img_cw)
        #plt.show()
        lines = cv2.HoughLines(img_cw, 1, np.pi / 180, 150, None, 0, 0)
        #img_lines = cv2.cvtColor(warped_img.copy(), cv2.COLOR_GRAY2RGB)
        img_lines = img_c
        lines_v = []
        lines_h = [] 
        T = []
        poly = Polygon(corners_w)
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
                cv2.line(img_dcw, pt1, pt2, (255,255,255), 1, cv2.LINE_AA)

                x,y = poly.exterior.xy
                ls = LineString([pt1, pt2])
                #plt.plot(x,y)
                #x,y = ls.coords.xy
                #plt.plot(x,y)
                #print(poly.crosses(ls))
                #plt.show()

                if poly.crosses(LineString([pt1, pt2])):
                    cv2.line(img_lines, pt1, pt2, (0,0,255), 1)
                else:
                    cv2.line(img_lines, pt1, pt2, (0,255, 0), 1)
                    print('skip')
                    continue
                cv2.imshow("PyGO", img_lines) 
                cv2.waitKey(1)
                    #continue

                if theta > np.pi:
                    theta -= np.pi
                if np.abs(theta) < np.pi/8:
                    lines_v.append([*pt1, *pt2])
                    #cv2.line(img_lines, pt1, pt2, (255,0,0), 1)
                else:
                    lines_h.append([*pt1, *pt2])
                    #cv2.line(img_lines, pt1, pt2, (0,255,0), 1)

        cv2.imshow("PyGO", img_lines) 
        cv2.waitKey(1000)
        lines = intersect(np.array(lines_v), np.array(lines_h))
        #[plt.axline((l[0],l[1]), (l[2],l[3])) for l in lines_v]
        #[plt.axline((l[0],l[1]), (l[2],l[3])) for l in lines_h]
        #plt.scatter(lines[:,0], lines[:,1])
        #plt.show(block=False)

 

        # init ref grid
        if len(lines) == 0:
            print('Could not find enought lines - calib failed')
            return 
        self.grid = get_ref_go_board_coords(np.min(lines, axis=0), np.max(lines, axis=0))
        # warp back to original images
        lines_raw_orig = cv2.perspectiveTransform(lines[:,None,:], np.linalg.inv(H)).squeeze()

        #pdb.set_trace()        
        #plt.scatter(lines[:,0], lines[:,1])
        #plt.imshow(img_dcw)
        #plt.show()

        #lines_warpedV = []
        #lines_warpedH = []

        #for lines in lines_vp:
        #    lines_warpedV.append(cv2.perspectiveTransform(np.stack(lines[0]).reshape(-1,1,2), H).reshape(-1,4))
        #    lines_warpedH.append(cv2.perspectiveTransform(np.stack(lines[1]).reshape(-1,1,2), H).reshape(-1,4))

        #lines_raw_orig = cluster(lines[0], lines[1])
        #lines = cluster(lines_warpedH[0], lines_warpedV[0])
        #lines = clear(lines)


        #img_max = (img_cw.shape[1], img_cw.shape[0])
        #img_min = (0,0)
    
        #img_cw = cv2.warpPerspective(img, H, img_limits)

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
                                'tolerance': 0.0001,
                                })
        #ty, param = reg.register(callback)
        ty, param = reg.register()

        R = np.eye(3)
        R[0:2,0:2]=param[0]
        R[0:2,2]=param[1]


        w_board = cv2.transform(np.array([self.grid]), (R))[0][:,:2]
        ow_board = cv2.perspectiveTransform(np.array([self.grid]), R@np.linalg.inv(H))[0]

        img_grid = plot_grid(img.copy(), ow_board.reshape(-1,2))
        #cv2.imshow('PyGO', img_grid)
        #cv2.waitKey(500)        
    
        src_pt = find_src_pt(ow_board, lines_raw_orig)
        H_refined = cv2.findHomography(src_pt, ow_board)[0]
        w_board = cv2.perspectiveTransform(np.array([self.grid]), R @ np.linalg.inv(H) @ np.linalg.inv(H_refined))[0]

        H_refined = R @ np.linalg.inv(H) @ np.linalg.inv(H_refined)

        img_grid = plot_grid(img.copy(), w_board.reshape(-1,2))
        cv2.imshow('PyGO', img_grid)
        cv2.waitKey(1)       
        self.H = H_refined 
        if self.check_patches_are_centered(img):
            self.hasEstimate=True
            print(self.H)
        else:
            self.H = np.eye(3)
            print('Calibration failed!')


    def check_patches_are_centered(self, img):
        cropped = self.extract(img)
        patches = self.imgToPatches(cropped)
        #cv2.imshow('PyGO', cropped)
        #cv2.waitKey(1000)
        if patches is None:
            #pdb.set_trace()
            print('could not extract patches')
            return False

        lines_x = []
        lines_y = []
        for i, patch in enumerate(patches):
            x,y = np.unravel_index(i, (19,19))
            if  x in [0,18] or y in [0,18]:
                #skip corners due to possible inclusion of board corners
                continue
            if x in [2,4,8,10,14,16] and y in [3,9,15]:
                #exclude calib patches
                continue
            thresh, patch_bw = cv2.threshold(toByteImage(patch), \
                                    0, \
                                    255, \
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            lines_x.append(np.argmax(np.sum(patch_bw, 0)))
            lines_y.append(np.argmax(np.sum(patch_bw, 1)))
        
        if np.std(lines_x) > 2.8 or np.std(lines_y) > 2.8:
            print("std x: {}".format(np.std(lines_x)))
            print("std y: {}".format(np.std(lines_y)))
            return False
        else:
            return True

 
    def extract(self, img):
        img_w = cv2.warpPerspective(img, np.linalg.inv(self.H), self.img_limits)

        #cv2.imshow('PyGO', img_w) 
        #cv2.waitKey(100)

        img_c_trim, (x,y) = mask_board(img_w, self.grid)
        self.go_board_shifted = self.grid - np.array([x,y])
        
        #cv2.imshow('PyGO', img_c_trim) 
        #cv2.waitKey(100)

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
        '''
            + = empty
            B = Black
            W = White
            O = Ref points on board

            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + B O W + + + B O W + + + B O W + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + B O W + + + B O W + + + B O W + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + B O W + + + B O W + + + B O W + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
            + + + + + + + + + + + + + + + + + + + 
        '''
        points_w = np.array([[ 4, 3],
                             [ 4, 6],
                             [ 4, 9],
                             [ 4,12],
                             [ 4,15],
                             [10, 3],
                             [10, 6],
                             [10, 9],
                             [10, 12],
                             [10,15],
                             [16, 3],
                             [16, 6],
                             [16, 9],
                             [16,12],
                             [16,15]])
        points_b = np.array([[ 2 ,3],
                             [ 2 ,6],
                             [ 2 ,9],
                             [ 2,12],
                             [ 2,15],
                             [ 8 ,3],
                             [ 8 ,6],
                             [ 8 ,9],
                             [ 8,12],
                             [ 8,15],
                             [14 ,3],
                             [14 ,6],
                             [14 ,9],
                             [14,12],
                             [14,15]])

        corners = [np.array([0,0]),
                   np.array([0,18]),
                   np.array([18,0]),
                   np.array([18,18])]
        idx_ = np.arange(1,18)
        edges_t = [np.array([0,i]) for i in idx_]
        edges_l = [np.array([i,0]) for i in idx_]
        edges_r = [np.array([18,i]) for i in idx_]
        edges_b = [np.array([i,18]) for i in idx_]
        edges = edges_b + edges_l + edges_r + edges_t

        idx_corner = [np.ravel_multi_index(arr, (19,19)) for arr in corners]
        idx_edge   = [np.ravel_multi_index(arr, (19,19)) for arr in edges]


        idx_w = np.ravel_multi_index(points_w.T, (19,19))
        idx_b = np.ravel_multi_index(points_b.T, (19,19))
        idx_n = np.arange(19*19)
        idx_n = np.delete(idx_n, np.concatenate((idx_w, idx_b, idx_corner, idx_edge)))
        patches = [[],[],[],[],[]]
        p = self.imgToPatches(img)
        for c, idx in enumerate([idx_w, idx_b, idx_n, idx_edge, idx_corner]):
            patches[c].append(np.array(p)[idx])
            #plt.imshow(np.vstack(np.array(p)[idx])/255)
            #plt.show()
        return patches
