import os
import cv2
import numpy as np
import pdb
from utils.image import toColorImage

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
                data.append(img)
                label.append(int(folder))
    return data, label


