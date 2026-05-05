import os
from re import I

import cv2
import imgaug.augmenters as iaa
import numpy as np
from joblib import dump, load
from skimage.feature import haar_like_feature, haar_like_feature_coord, hog
from skimage.filters import sobel
from skimage.transform import  integral_image
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm

from pygo.classifiers.BaseGoClassifier import Classifier
from pygo.utils.data import  load_training_data2
from pygo.utils.image import *

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

