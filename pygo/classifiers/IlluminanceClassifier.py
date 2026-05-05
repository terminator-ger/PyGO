import cv2
from pygo.classifiers.BaseGoClassifier import Classifier
from pygo.utils.image import *
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pygo.utils.data import  load_and_augment_training_data, weights_path

from joblib import load, dump

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

