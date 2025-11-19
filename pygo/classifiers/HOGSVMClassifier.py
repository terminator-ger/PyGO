import os
import numpy as np
import cv2

from joblib import load, dump

from skimage.feature import hog

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC

from pygo.utils.data import load_and_augment_training_data, weights_path
from pygo.classifiers.BaseGoClassifier import Classifier

class HOGSVMClassifier(Classifier):
    def __init__(self):
        self.hasWeights = False
        self._is_fitted  = True
            
    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted



    def predict_prob(self, patches):
        if not hasattr(self, 'clf'):
            self.load()
        HOG = np.asarray([self.extract_feature_image(patch) for patch in patches])
        result = self.clf.predict_proba(HOG)
        result = result.reshape(19,19, 3).T.reshape(-1, 3)
        return result


    def predict(self, patches):
        HOG = np.asarray([self.extract_feature_image(patch) for patch in patches])
        result = self.clf.predict(HOG)
        result = result.reshape(19,19).T.reshape(-1)
        return result


    def extract_feature_image(self, patches):
        patches = cv2.resize(patches, (32,32))
        fd = hog(patches, orientations=8, pixels_per_cell=(8,8),
                    cells_per_block=(1,1), visualize=False, channel_axis=-1)
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
        weights_file = weights_path("weights", "hogsvm.joblib")
        if os.path.exists(weights_file):
            self.clf = load(weights_file)
            self.hasWeights = True
        else:
            print('Failed to Restore HOGSVG Classification Alg')
            self.hasWeights = False

    def store(self):
        weights_file = weights_path("weights", "hogsvm.joblib")
        dump(self.clf , weights_file)