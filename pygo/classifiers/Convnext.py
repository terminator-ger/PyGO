import os
import torch as th
import numpy as np

from pygo.utils.data import load_and_augment_training_data, weights_path
from pygo.GoNet import GoNet
from pygo.classifiers.BaseGoClassifier import Classifier
from timm import create_model

class ConvnextClassifier(Classifier):
    _parameter_constraints = {
        "weights_file": [str],
        "num_classes": [int],
    }
     
    def __init__(self, weights_file, classes=3) -> None:
        self.classes_ = classes
        self.weights_file = weights_file
        self._is_fitted = True
            
    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def predict(self, patches):
        if not hasattr(self, 'model'):
            self.model = create_model('convnext_nano.in12k', pretrained=False, num_classes=3, )
            self.model.eval()
            self.load()

        x = th.from_numpy(np.array(patches).astype(np.float32)).permute(0,3,1,2)
        if th.max(x) > 1.0:
            x = x / 255
        lbl = self.model(x)
        lbl = lbl.detach().cpu().numpy()
        lbl = np.argmax(lbl, axis=1)

        for size in [9, 13, 19]:
            if len(patches) == size*size:
                lbl = lbl.reshape(size, size, self.classes_)
                lbl = np.rot90(np.fliplr(lbl))
                break
 
        lbl = lbl.reshape(-1)
        return lbl


    def predict_proba(self, patches):
        if not hasattr(self, 'model'):
            self.model = create_model('convnext_nano.in12k', pretrained=True, num_classes=3, )
            self.model.eval()
            self.load()


        x = th.from_numpy(np.array(patches).astype(np.float32)).permute(0,3,1,2)
        if th.max(x) > 1.0:
            x = x / 255
        lbl = self.model(x)
        lbl = lbl.detach().cpu().numpy()
        
        for size in [9, 13, 19]:
            if len(patches) == size*size:
                lbl = lbl.reshape(size, size, self.classes_)
                lbl = np.rot90(np.fliplr(lbl))
                break
 
        lbl = lbl.reshape(-1, self.classes_)
 
        return lbl


    def load(self):
        weights_file = weights_path("weights", self.weights_file)
        if os.path.exists(weights_file):
            self.model.load_state_dict(th.load(weights_file, weights_only=True))
        else:
            print('Failed to Restore ConvGO Classification Alg')

    def store(self):
        weights_file = weights_path("weights", self.weights_file)
        th.save(self.model, weights_file)