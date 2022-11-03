import cv2
cv2.namedWindow("", 0)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pygo.utils.color import *
from pygo.utils.image import *
from scipy.ndimage import label
import pdb


def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=1)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, 2, 3)
    #dt = cv2.normalize(dt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    height, width = img.shape
    stone_size = min(height/19, width/19)
    print(stone_size/2)
    _, dt = cv2.threshold(dt, stone_size*0.4, dt.max(), cv2.THRESH_BINARY)

    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now. 
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl


def create_mask_white(ipt):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    mask_white = cv2.equalizeHist(ipt)
    mask_white[mask_white>25] = 255
    mask_white[mask_white<15] = 0
    mask_white =  cv2.adaptiveThreshold(mask_white, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 
                                    31, 1)
    mask_white = cv2.erode(mask_white, kernel, iterations=5)
    mask_white = cv2.dilate(mask_white, kernel, iterations=5)
    return mask_white


def remove_glare2(img):
    clahefilter = cv2.createCLAHE(clipLimit=2,
                    tileGridSize = (15,15))
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





if __name__ == '__main__':

    imgs = [cv2.imread('out{}.png'.format(i)) for i in range(6)]
    img = imgs[2]
    img = remove_glare2(img)
    img_c = img.copy()
    hsv = toHSVImage(  img.copy())
    yuv = toYUVImage(  img.copy())
    cmyk = toCMYKImage(img.copy())
    cmyk1, cmyk2, cmyk3, cmyk4 = cv2.split(cmyk)
    hsv1, hsv2, hsv3 = cv2.split(hsv)

    # Otsu's thresholding after Gaussian filtering
    mm = np.zeros_like(hsv2, dtype=np.uint8)
    blur = cv2.GaussianBlur(yuv[:,:,2],(5,5),0)
    ret3,mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)
    i2 = img.copy()
    i2[mask==0] = 0
    mm[mask==255] = 255

    white = hsv2
    mask_white = create_mask_white(white)

    #plt.imshow(mask_white)
    #plt.show()
    i2[mask_white==255] = 0
    mm[mask_white==255] = 0 
    #plt.imshow(i2)
    #plt.show()

    #plt.imshow(mm)
    #plt.show()


    markers_black = segment_on_dt(img, mask)
    plt.imshow(markers_black)
    plt.show()
    hsv1[mm==0] = 0
    hsv2[mm==0] = 0
    cmyk4[mm==0] = 0
    cmyk4[mm==0] = 0
    plt.subplot(221)
    plt.imshow(hsv1)
    plt.subplot(222)
    plt.imshow(hsv2)
    plt.subplot(223)
    plt.imshow(cmyk3)
    plt.subplot(224)
    plt.imshow(cmyk4)
    plt.show()

    blur = cv2.GaussianBlur(cmyk4,(3,3),0)
    mask_cmyk = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    mask_cmyk = np.logical_xor(mask_cmyk, mm).astype(np.uint8)

    plt.imshow(np.dstack((cmyk4, mask_cmyk*255, cmyk4)))
    plt.show()


    blur = cv2.GaussianBlur(hsv2,(3,3),0)
    mask_hsv1 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 3, -1)
    
    plt.imshow(np.dstack((hsv2, mask_hsv1*255, hsv2)))
    plt.show()


    #mask_hsv1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    mask_hsv1 = np.logical_and(mask_hsv1, mm).astype(np.uint8)

    mask = np.logical_or(mask_cmyk, mask_hsv1).astype(np.uint8)
    black_areas = np.logical_not(np.logical_or(mask, np.logical_not(mm))).astype(np.bool)
    bright_black_areas = np.logical_and(np.logical_not(black_areas), mm).astype(np.bool)
    kernel_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bright_black_areas = cv2.dilate(bright_black_areas.astype(np.uint8), kernel_ell)
    gray =toGrayImage(img)
    #plt.subplot(121)
    #plt.imshow(np.dstack((gray, 255*black_areas, gray)))
    #plt.subplot(122)
    #plt.imshow(np.dstack((gray, 255*bright_black_areas, gray)))
    #plt.show()

    hsv1, hsv2, hsv3 = cv2.split(hsv)

    med_color_1 = np.median(hsv1[black_areas])
    med_color_2 = np.median(hsv2[black_areas])
    med_color_3 = np.median(hsv3[black_areas])
    #hsv1[bright_black_areas] = med_color_1
    #hsv2[np.logical_not(black_areas)] = med_color_2
    #hsv3[np.logical_not(black_areas)] = med_color_3
    hsv2 = cv2.inpaint(hsv2, bright_black_areas, 5, cv2.INPAINT_NS)
    #plt.imshow(hsv2)
    #plt.show()
    hsv = cv2.merge([hsv1, hsv2, hsv3])
    #plt.subplot(121)
    #plt.imshow(img)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    plt.subplot(122)
    plt.imshow(img)
    plt.show()
    width, height, _ = img.shape
    est_size = int(max(width//19, height//19))//4
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (est_size, est_size))
    black_areas = (255*black_areas).astype(np.uint8)
    black_areas2 = cv2.dilate(black_areas, kernel)

    markers_black = segment_on_dt(img, black_areas2)
    plt.imshow(markers_black)
    plt.show()
    pdb.set_trace()