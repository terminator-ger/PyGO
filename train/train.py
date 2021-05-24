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
from GoNet import GoNet
import torch as th
import torch.nn.functional as F

import warnings
warnings.filterwarnings('always') 
def toNP(x):
    return x.detach().cpu().numpy()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


if __name__ == '__main__':

    classes = [0,1,2]
    data = []
    label = []

    for c in classes:
        for file in os.listdir('{}'.format(c)):
            d = cv2.imread('{}/{}'.format(c,file))
            d = cv2.resize(d, (32,32))
            data.append(d)
            label.append(c)


    data = np.array(data)
    y = np.array(label)
    y_cat = np.array(label)

    sometimes = lambda aug: iaa.Sometimes(0.85, aug)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        iaa.contrast.GammaContrast(),
        iaa.imgcorruptlike.Saturate(),
        iaa.imgcorruptlike.Brightness(),
        #iaa.imgcorruptlike.SpeckleNoise()
    ])
    data = seq(images=data)
#
#    x_train = []
#    for d in data:
#        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
#        d = img_as_float(d)
#
#        x_train.append(get_feat_vec(d))
#   
#    X = np.array(x_train)

    X = np.array(data)
    y = to_categorical(y, 3)


    idx = np.arange(len(data))
    np.random.shuffle(idx)

    data = data[idx]
    L = len(data)
    SPLT = 100
    X_train = X[idx[:L-SPLT]]
    y_train = y[idx[:L-SPLT]]
    X_test  = X[idx[-SPLT:]]
    y_test  = y[idx[-SPLT:]]
    
#    model = SVC()
#    params = {'C': np.linspace(1e4, 1e-4),
#              'kernel': ['rbf'],
#              }
#    search = GridSearchCV(model, params)
#    search.fit(X_train, y_train)
#    print(search.best_params_)
#    best = search.best_estimator_
#    y_pred = best.predict(X_test)
#
#    print(classification_report(y_test, y_pred))
    X_train = th.from_numpy(X_train)
    y_train = th.from_numpy(y_train)
    X_test  = th.from_numpy(X_test)
    X_train = X_train.permute(0,3,1,2)/255.0
    X_test  = X_test.permute(0,3,1,2)/255.0
    y_test = y_cat[idx[-SPLT:]]

    #model = GoNet()
    model = th.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)



    opt = th.optim.Adam(model.parameters())
    loss_fn = th.nn.MultiLabelSoftMarginLoss()
    for i in range(400):
        opt.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        opt.step()
        with th.no_grad():
            y_pred = model(X_test)
            y_pred = toNP(y_pred)
            y_pred = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_test, y_pred, average='micro')
        if i % 5 == 0:
            th.save(model, 'mobile_weights_{}_{}.pt'.format(i, f1))
            print(classification_report(y_test, y_pred))

            plt.subplot(131)
            if np.any((y_pred==0)):
                img = np.vstack(data[-SPLT:][y_pred==0])
                plt.imshow(img)
            plt.subplot(132)
            if np.any((y_pred==1)):
                plt.imshow(np.vstack(data[-SPLT:][y_pred==1]))
            plt.subplot(133)
            if np.any((y_pred==2)):
                plt.imshow(np.vstack(data[-SPLT:][y_pred==2]))
            plt.show(block=False)
            plt.pause(0.01)


#    dump(best, 'svm.joblib') 