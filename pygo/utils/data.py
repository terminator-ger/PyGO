import os
import cv2
import numpy as np
import pdb
from pygo.utils.image import toCMYKImage, toColorImage, toDoubleImage
import imgaug.augmenters as iaa
from tqdm import tqdm
from joblib import load
import importlib

def weights_path(module: str, name: str) -> str:
    return importlib.resources.files(module).joinpath(name)

def load_training_data_old(classes):
    x = []
    y = []

    for folder in os.listdir('data'):
        if int(folder) in classes:
            for file in os.listdir(os.path.join('data', folder)):
                #img = io.imread(os.path.join('data', folder, file))
                img = cv2.imread(os.path.join('data', folder, file))
                img = toColorImage(img)
                img = cv2.resize(img, (32,32))
                img = np.array(img)
                x.append(img)
                f = int(folder.split('.png')[0])
                y.append(f)

    return x, y

def save_training_data(patches):
    for c, ptch in enumerate(patches):
        for i, p in enumerate(ptch[0]):
            files = os.listdir(os.path.join('data','{}'.format(c)))
            if len(files)>0:
                files = [int(x.strip('.png')) for x in files]
                max_n = max(files)+1
            else:
                max_n = 0
            cv2.imwrite(os.path.join('data', '{}'.format(c), '{}.png'.format(max_n+i)), p)

def load_training_data():
    patches_arr = [[],[],[],[],[]]
    for folder in os.listdir('data'):
            for file in os.listdir(os.path.join('data', folder)):
                img = cv2.imread(os.path.join('data', folder, file))
                patches_arr[int(folder)].append(img)
    return patches_arr


def load_training_data2():
    data = []
    label = []
    for folder in os.listdir('data'):
            for file in os.listdir(os.path.join('data', folder)):
                img = cv2.imread(os.path.join('data', folder, file))
                img = toCMYKImage(img)[:,:,3]
                img = toColorImage(img)
                data.append(img)
                label.append(int(folder))
    return data, label


def load_and_augment_training_data(feat_fn):
    x_train = []
    y_train = []
 
   # split patches back into their categories
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 20% of all images
        iaa.Resize((35,35)),
        iaa.CropToFixedSize(width=32, height=32),
        #iaa.color.MultiplyAndAddToBrightness(),
        #iaa.color.MultiplyBrightness(mul=(0.8,1.2))
        iaa.Multiply((0.8, 1.2), per_channel=0.2)
    ])
        
    patches_arr = load_training_data()

    #inflate stone samples
    def inflate(cls, times):
        for i in range(times):
            for c in cls:
                for p in patches_arr[c]:
                    p = seq(image=p)
                    p = toDoubleImage(np.array(p))
                    x_train.append(p)
                    y_train.append(c)

    inflate([0,1], 5)#13)
    inflate([2], 1)
    inflate([3], 3)
    inflate([4], 5)#62)

    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    x = [x_train[i] for i in idx]
    y = [y_train[i] for i in idx]
    X = []
    for img in tqdm(x):
        X.append(feat_fn(img))

    samples = len(x)
    SPLT = int(0.1*samples)

    #group 
    X_train = np.array(X[:-SPLT])
    y_train = np.array(y[:-SPLT])
    X_test  = np.array(X[-SPLT:])
    y_test  = np.array(y[-SPLT:])

    def replace(vec, what, wth):
        vec[vec==what] = wth
        return vec

    y_train = replace(y_train, 0, 1)
    y_train = replace(y_train, 1, 1)
    y_train = replace(y_train, 2, 0)
    y_train = replace(y_train, 3, 0)
    y_train = replace(y_train, 4, 0)

    y_test = replace(y_test, 0, 1)
    y_test = replace(y_test, 1, 1)
    y_test = replace(y_test, 2, 0)
    y_test = replace(y_test, 3, 0)
    y_test = replace(y_test, 4, 0)


    print('No Train Samples: {}'.format(len(X_train)))
    print('No Test Samples: {}'.format(len(X_test)))
 
    return X_train, y_train, X_test, y_test

