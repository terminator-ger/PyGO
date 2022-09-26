from numpy.lib.shape_base import expand_dims
import sklearn
import cv2
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
from feature import get_feat_vec
import torch as th
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from scipy.distance import distance_matrix

import warnings
warnings.filterwarnings('always') 
def toNP(x):
    return x.detach().cpu().numpy()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


if __name__ == '__main__':

    classes = [0,1,2]
    channels = 1
    data = []
    label = []


    #model = torchvision.models.mobilenet_v2(pretrained=True)
    #model.features[0] = th.nn.Conv2d(1, 32, (3,3), stride=(2,2), padding=(1,1), bias=False)
    #model.classifier[1] = th.nn.Linear(1280, 3)
    #pdb.set_trace()
    #model = th.load('train/mobile_weights_5_1.0.pt')
    from sklearn.cluster import KMeans
    extractor = cv2.SIFT_create()
    km = KMeans(3)
    BOW = []
    for c in classes:        
        c_train = []
        for file in os.listdir('{}'.format(c)):
            d = cv2.imread('{}/{}'.format(c,file))
            d = cv2.resize(d, (32,32))
            _, feat = extractor.detectAndCompute(d, None)
            c_train.append(feat)

            data.append(d)
            label.append(c)
        # reduce
        km.fit(np.stack(c_train))
        BOW.append(km.cluster_centers_)


    def descToHist(desc):
        D = distance_matrix(desc, BOW)
        nn = np.argmin(D, axis=0)
        _, hist = np.unique(nn, count=True)
        return hist

    x_train = []
    for c in classes:        
        c_train = []
        for file in os.listdir('{}'.format(c)):
            d = cv2.imread('{}/{}'.format(c,file))
            d = cv2.resize(d, (32,32))
            _, feat = extractor.detectAndCompute(d, None)
            h = descToHist(feat)
            x_train.append(h)
 
    X = np.array(X_train)
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(X,y)


    dump(model, 'knn.joblib') 
