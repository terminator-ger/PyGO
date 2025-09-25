import numpy as np
from re import I
from turtle import back

from requests import patch
import pdb
from pudb import set_trace
import logging
import numpy as np
import matplotlib.pyplot as plt
from joblib import load, dump
from pygo.Signals import OnBoardGridSizeKnown, Signals
from skimage import exposure
from scipy.spatial import distance_matrix
from pygo.GoNet import GoNet
import torch as th
import cv2
import os
import imgaug.augmenters as iaa
import torch as th
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple
from skimage import (
    data, restoration, util
)
from scipy.ndimage import label

from skimage.transform import integral_image
from skimage.feature import haar_like_feature, hog, haar_like_feature_coord

from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.filters import sobel

from skimage.draw import circle_perimeter
from skimage.transform import hough_circle, hough_circle_peaks
 

import warnings

from pygo.utils.line import point_in_circle
from pygo.utils.debug import Timing
from pygo.utils.image import *
from pygo.utils.data import load_training_data, save_training_data, load_training_data2, load_and_augment_training_data, weights_path
from pygo.utils.debug import DebugInfo, DebugInfoProvider
from pygo.utils.typing import B1CImage, B3CImage, GoBoardClassification
from pygo.GoBoard import GoBoard

class Classifier:
    def predict(self, patches):
        raise NotImplementedError()

    def train(self, patches):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def store(self):
        raise NotImplementedError()

def sliding_window(image, stepSize, windowSize):
    #trim img
    dx = image.shape[0] // windowSize[0]
    dy = image.shape[1] // windowSize[1]
    image = image[:dx*windowSize[0], :dy*windowSize[1]]
    for y in range(0, image.shape[0]-windowSize[0], stepSize):
        for x in range(0, image.shape[1]-windowSize[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

class HOGSVMClassifier(Classifier):
    def __init__(self):
        self.hasWeights = False
        self.clf = None
        self.load()


    def predict(self, patches):
        result = np.zeros((19*19))
        for i, patch in enumerate(patches):
            HOG = self.extract_feature_image(patch)
            P = self.clf.predict(HOG.reshape(1,-1))
            result[i] = P
        result = result.reshape(19,19).T.reshape(-1)
        return result


    def extract_feature_image(self, patches):
        patches = cv2.resize(patches, (32,32))
        fd = hog(patches, orientations=8, pixels_per_cell=(8,8),
                    cells_per_block=(1,1), visualize=False, channel_axis=-1)
        #cv2.imshow('img', img) 
        #cv2.waitKey(400)

        return fd


    def train(self):
        X_train, y_train, X_test, y_test = load_and_augment_training_data(self.extract_feature_image)

        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
       
        self.clf.fit(X_train, y_train)

        y_pred = self.clf.predict(X_test)

        print(classification_report(y_test, y_pred))

        self.hasWeights = True
        self.store()




    def load(self):
        weights_file = weights_path("pygo.weights", "hogsvm.joblib")
        if os.path.exists(weights_file):
            self.clf = load(weights_file)
            self.hasWeights = True
        else:
            print('Failed to Restore HOGSVG Classification Alg')
            self.hasWeights = False

    def store(self):
        dump(self.clf , 'weights/hogsvm.joblib')



class IlluminanceClassifier(Classifier):
    def __init__(self) -> None:
        self.hasWeights = False
        self.clf = None
        self.load()

    def extract_feature_image(self, img):
        """Extract the haar feature for the current image"""
        # extract only the lightness part
        if len(img.shape) == 2:
            # gray to color
            img = cv2.cvtColor(toByteImage(img), cv2.COLOR_GRAY2RGB)
        img = toByteImage(img)
        img = toCMYKImage(img)[:,:,3]
        # features for black
        thresh, img_bw = cv2.threshold(img, \
                                    0, \
                                    255, \
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        sum_b = np.sum(img_bw)
        img_x = np.mean(img_bw, axis=0)
        img_y = np.mean(img_bw, axis=1)

        #f0 = np.histogram(img_x, bins=2, range=(0,255))[1]
        #f1 = np.histogram(img_y, bins=2, range=(0,255))[1]

        #for white
        thresh, img_ww = cv2.threshold(img, \
                                    0, \
                                    255, \
                                    cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        img_wx = np.mean(img_ww, axis=0)
        img_wy = np.mean(img_ww, axis=1)
        sum_w = np.sum(img_ww)

        #f2 = np.histogram(img_wx, bins=2, range=(0,255))[1]
        #f3 = np.histogram(img_wy, bins=2, range=(0,255))[1]
        return np.concatenate((img_x, img_y,img_wx, img_wy, np.array([sum_w]), np.array([sum_b])))
        return np.array([sum_w, sum_b])

    def predict(self, patches):
        x = []
        for i in (range(len(patches))):
            x.append(self.extract_feature_image(patches[i]))
        lbl = self.clf.predict(x)
        return lbl

    def predict_prob(self, patches):
        x = []
        for i in (range(len(patches))):
            x.append(self.extract_feature_image(patches[i]))
        lbl = self.clf.predict_proba(x)
        return lbl


    def train(self):
        X_train, y_train, X_test, y_test = load_and_augment_training_data(self.extract_feature_image)

        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
       
        self.clf.fit(X_train, y_train)

        y_pred = self.clf.predict(X_test)

        print(classification_report(y_test, y_pred))

        self.hasWeights = True
        self.store()




    def load(self):
        if os.path.exists('illuminance.joblib'):
            self.clf = load('illuminance.joblib')
            self.hasWeights = True
        else:
            print('Failed to Restore Illuminance Classification Alg')
            self.hasWeights = False

    def store(self):
        dump(self.clf , 'illuminance.joblib')


class HaarClassifier(Classifier):
    def __init__(self) -> None:
        self.hasWeights = False
        self.clf = None
        self.feature_type_sel = None
        self.feature_coord_sel = None
        self.load()

    def extract_feature_image(self, img, feature_type, feature_coord=None):
        """Extract the haar feature for the current image"""
        img = cv2.resize(img, (32,32))
        ii = integral_image(img)
        ret = haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                                feature_type=feature_type,
                                feature_coord=feature_coord)
        return ret

    def predict(self, patches):
        x = []
        for i in (range(len(patches))):
            x.append(self.extract_feature_image(cv2.cvtColor(patches[i], cv2.COLOR_RGB2GRAY), 
                                            self.feature_type_sel, 
                                            self.feature_coord_sel))
        lbl = self.clf.predict(x)
        # replace corners
        return lbl

    def predict_prob(self, patches):
        x = []
        for i in (range(len(patches))):
            x.append(self.extract_feature_image(cv2.cvtColor(patches[i], cv2.COLOR_RGB2GRAY), 
                                            self.feature_type_sel, 
                                            self.feature_coord_sel))
        lbl = self.clf.predict_proba(x)
        return lbl



    def train(self):

        #X_train, y_train, X_test, y_test = load_and_augment_training_data(self.extract_feature_image)
        x_train = []
        y_train = []

        data, label = load_training_data2()
        patches = [[],[],[],[],[]]
        for lbl, img in zip(label, data):
            patches[lbl].append(img)


        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 20% of all images
            #iaa.imgcorruptlike.Brightness((1,3)),
            #iaa.color.MultiplyAndAddToBrightness(),
            iaa.color.MultiplyBrightness((0.9, 1.1))
        ])
        
        patches_arr = [[],[],[],[],[]]
        for c in range(len(patches)):
            for i in range(len(patches[c])):
                #patches_arr[c].append(cv2.cvtColor(np.array(patches[c][i]).astype(np.uint8), cv2.COLOR_GRAY2RGB))
                patches_arr[c].append(toByteImage(patches[c][i]))


        #inflate stone samples
        for i in range(5):
            patches_mod = [[],[],[],[],[]]
            for c in [0,1,3,4]:
                patches_mod[c] = seq(images=patches_arr[c])
                for p in patches_mod[c]:
                    #p = seq(image=p)
                    p = toDoubleImage(np.array(p))
                    x_train.append(p)
                    y_train.append(c)

        c = 2
        patches_mod[c] = seq(images=patches_arr[c])
        for p in patches_mod[c]:
            #p = seq(image=p)
            p = toDoubleImage(np.array(p))
            x_train.append(p)
            y_train.append(c)
 

 
        x_train = [cv2.cvtColor((x*255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for x in x_train]
        idx = np.arange(len(x_train))
        np.random.shuffle(idx)
        x = [x_train[i] for i in idx]
        y = [y_train[i] for i in idx]

        feature_types = ['type-2-x', 'type-2-y']
        X = []
        for img in tqdm(x):
            X.append(self.extract_feature_image(img, feature_types))


        ## Extract all possible features
        #feature_coord, feature_type = \
        #haar_like_feature_coord(width=X.shape[2], height=X.shape[1],
        #                        feature_type=feature_types) 

        # Compute the result

        SPLT = 100
        X_train = X[:-SPLT]
        y_train = y[:-SPLT]
        X_test  = X[-SPLT:]
        y_test  = y[-SPLT:]

        print('No Train Samples: {}'.format(len(X_train)))
        print('No Test Samples: {}'.format(len(X_test)))

        # Train a random forest classifier and assess its performance
        self.clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=100, n_jobs=-1, random_state=0)
        self.clf.fit(X_train, y_train)



        idx_sorted = np.argsort(self.clf.feature_importances_)[::-1]
        cdf_feature_importances = np.cumsum(self.clf.feature_importances_[idx_sorted])
        cdf_feature_importances /= cdf_feature_importances[-1]  # divide by max value
        sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.7)
        sig_feature_percent = round(sig_feature_count /
                                    len(cdf_feature_importances) * 100, 1)

        # Extract all possible features
        feature_coord, feature_type = \
            haar_like_feature_coord(width=x[0].shape[0], 
                                    height=x[0].shape[1],
                                    feature_type=feature_types)

        self.feature_coord_sel = feature_coord[idx_sorted[:sig_feature_count]]
        self.feature_type_sel = feature_type[idx_sorted[:sig_feature_count]]
        X = []
        for img in tqdm(x):
            X.append(self.extract_feature_image(img, 
                                            self.feature_type_sel,
                                            self.feature_coord_sel))


        SPLT = 100
        X_train = X[:-SPLT]
        y_train = y[:-SPLT]
        X_test  = X[-SPLT:]
        y_test  = y[-SPLT:]


        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        print(classification_report(y_test, y_pred))

        self.hasWeights = True
        self.store()

    def load(self):
        import pickle
        if os.path.exists('weights/haar.pkl'):
            with open('weights/haar.pkl', 'rb') as file:
                self.clf = pickle.load(file)
            with open('weights/feat_coord.pkl', 'rb') as file:
                self.feature_coord_sel = pickle.load(file)
            with open('weights/feat_type.pkl', 'rb') as file:
                self.feature_type_sel = pickle.load(file)
            self.hasWeights = True

        #if os.path.exists('weights/haar.joblib'):
        #    self.clf = load('weights/haar.joblib')
        #    self.feature_coord_sel = load('weights/feat_coord.joblib')
        #    self.feature_type_sel = load('weights/feat_type.joblib')
        #    self.hasWeights = True
        else:
            print('Failed to Restore Haar Classification Alg')
            self.hasWeights = False

    def store(self):
        pass
        #dump(self.clf , 'haar.joblib')
        #dump(self.feature_coord_sel, 'weights/feat_coord.joblib')
        #dump(self.feature_type_sel , 'weights/feat_type.joblib')

class GoClassifier(Classifier):
    def __init__(self) -> None:
        self.hasWeights = False
        self.num_classes = 5
        self.model = GoNet(num_classes=self.num_classes)
        self.load()

    def predict(self, patches):
        #x = []
        #for i in range(len(patches)):
        #    x.append(cv2.cvtColor(np.array(patches[i]).astype(np.uint8), cv2.COLOR_GRAY2RGB))
        x = th.from_numpy(np.array(patches).astype(np.float32)).permute(0,3,1,2)
        lbl = self.model(x)
        lbl = lbl.detach().cpu().numpy()
        lbl = np.argmax(lbl, axis=1)

        # replace corners
        lbl[lbl==3] = 2
        lbl[lbl==4] = 2
        return lbl


    def predict_prob(self, patches):
        #x = []
        #for i in range(len(patches)):
        #    x.append(cv2.cvtColor(np.array(patches[i]).astype(np.uint8), cv2.COLOR_GRAY2RGB))
        x = th.from_numpy(np.array(patches).astype(np.float32)).permute(0,3,1,2)
        lbl = self.model(x)
        lbl = lbl.detach().cpu().numpy()
        # replace corners
        return lbl



    def train(self):
        self.model = GoNet(num_classes=self.num_classes)
        X_train, y_train, X_test, y_test = load_and_augment_training_data((lambda x:x))
       
        X_train = th.from_numpy(X_train.astype(np.float32))
        y_train = th.from_numpy(y_train.astype(np.int_))
        X_test  = th.from_numpy(X_test.astype(np.float32))
        y_test  = th.from_numpy(y_test.astype(np.int_))
        # channels to pos 1
        X_train = X_train.permute(0,3,1,2)
        X_test  = X_test.permute(0,3,1,2)

        batch_size = X_train.size()[0]
        opt = th.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.005)
        loss_fn = th.nn.CrossEntropyLoss()
        print('train')
        for i in range(40):
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
                for cls in range(self.num_classes):
                    plots=101+self.num_classes*10
                    plt.subplot(plots+cls)
                    if np.any((y_pred==cls)):
                        plt.imshow(np.vstack(X_test[y_pred==cls, 0]))

                plt.savefig('{}.png'.format(i), dpi=400)

        print(classification_report(y_test, y_pred))
        self.hasWeights = True

    def load(self):
        if os.path.exists('weights.pt'):
            self.model = th.load('weights.pt')
            self.hasWeights = True
        else:
            print('Failed to Restore ConvGO Classification Alg')
            self.hasWeights = False

    def store(self):
        th.save(self.model, 'weights.pt')

class KNNClassifier(Classifier):
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
            print('Failed to Restore KNN Classification Alg')
            self.hasWeights = False

    def store(self):
        dump(self.model, 'knnc.joblib') 

class BOWClassifier(Classifier):
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
            print('Failed to Restore BOW Classification Alg')
            self.hasWeights = False

    def store(self):
        dump(self.model, 'knn.joblib') 
        dump(self.BOW, 'bow.joblib') 

class HaarCascadeClassifier(Classifier):
    def __init__(self):
        self.cls = cv2.CascadeClassifier()
        self.load()

    def train(self):
        pass
    
    def load(self):
        self.cls.load('weights/cascade2.xml')

    def predict(self, img):
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        stones = self.cls.detectMultiScale(frame_gray, 
                                            scaleFactor=1.3, 
                                            minSize=(11,11), 
                                            maxSize=(32,32))
        for (x,y,w,h) in stones:
            center = (x + w//2, y + h//2)
            img = cv2.ellipse(img, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        cv2.imshow('Capture - Face detection', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    cls = HOGSVMClassifier()
    cls.train()

    #for i in range(1,8):
    #    img = cv2.imread('debug/{}.png'.format(i))
    #    result = cls.predict(img)
    #    plt.subplot(121)
    #    plt.imshow(result)
    #    plt.subplot(122)
    #    plt.imshow(img)
    #    plt.show() 

















