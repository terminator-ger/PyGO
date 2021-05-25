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
from GoNet import GoNet
import torch as th
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

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

    model = GoNet()
    #model = torchvision.models.mobilenet_v2(pretrained=True)
    #model.features[0] = th.nn.Conv2d(1, 32, (3,3), stride=(2,2), padding=(1,1), bias=False)
    #model.classifier[1] = th.nn.Linear(1280, 3)
    #pdb.set_trace()
    #model = th.load('train/mobile_weights_5_1.0.pt')

    for c in classes:
        for file in os.listdir('{}'.format(c)):
            d = cv2.imread('{}/{}'.format(c,file))
            d = cv2.resize(d, (32,32))
            data.append(d)
            label.append(c)

    sometimes = lambda aug: iaa.Sometimes(0.85, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        iaa.contrast.GammaContrast(),
        iaa.imgcorruptlike.Saturate(),
        iaa.imgcorruptlike.Brightness(),
    ])
    data = seq(images=data)

    if channels == 1:
        data = [np.expand_dims(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), -1) for x  in data]
    data = [x/255.0 for x in data]
#    x_train = []
#    for d in data:
#        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
#        d = img_as_float(d)
#        x_train.append(get_feat_vec(d))
#    X = np.array(x_train)

    idx = np.arange(len(data))
    np.random.shuffle(idx)


    X  = np.array(data)
    y  = np.array(label)
    X = X[idx]
    y = y[idx]

    SPLT = 100
    X_train = X[:-SPLT]
    y_train = y[:-SPLT]
    X_test  = X[-SPLT:]
    y_test  = y[-SPLT:]
    
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

    X_train = th.from_numpy(X_train.astype(np.float32))
    y_train = th.from_numpy(y_train.astype(np.int_))
    X_test  = th.from_numpy(X_test.astype(np.float32))
    y_test  = th.from_numpy(y_test.astype(np.int_))

    # channels to pos 1
    X_train = X_train.permute(0,3,1,2)
    X_test  = X_test.permute(0,3,1,2)

    batch_size = X_train.size()[0]
    opt = th.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = th.nn.CrossEntropyLoss()
    print('train')
    for i in range(400):
        #permutation = th.randperm(X_train.size()[0])
        permutation = np.arange(X_train.size()[0])
        
        for j in range(0, X_train.size()[0], batch_size):
            indices = permutation[j:j+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        with th.no_grad():
            y_pred = model(X_test)
            loss_test = loss_fn(y_pred, y_test)
            y_pred = F.log_softmax(y_pred, -1)
            y_pred = toNP(y_pred)
            y_pred = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_test, y_pred, average='micro')
            print("Epoch {} : {:0.2f}, {:0.3f}, {:0.3f}".format(i,f1, toNP(loss), toNP(loss_test)))

        if i % 5 == 0:
            th.save(model, 'weights_{}_{}.pt'.format(i, f1))
            plt.subplot(131)
            if np.any((y_pred==0)):
                plt.imshow(np.vstack(X_test[y_pred==0, 0]))
            plt.subplot(132)
            if np.any((y_pred==1)):
                plt.imshow(np.vstack(X_test[y_pred==1, 0]))
            plt.subplot(133)
            if np.any((y_pred==2)):
                plt.imshow(np.vstack(X_test[y_pred==2, 0]))
            plt.savefig('{}.png'.format(i), dpi=400)

    print(classification_report(y_test, y_pred))

#    dump(best, 'svm.joblib') 