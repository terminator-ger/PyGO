import cv2
import warnings
import numpy as np

from enum import  auto
from typing import Tuple, List
from dataclasses import dataclass
from pygo.Signals import OnBoardGridSizeKnown, CoreSignals
from pygo.utils.debug import Timing
from pygo.utils.image import *
from pygo.utils.debug import DebugInfoProvider, debugkeys
from pygo.utils.plot import Plot
from pygo.utils.typing import B3CImage, GoBoardClassification
from pygo.GoBoard import GoBoard
from pygo.classifiers.BaseGoClassifier import Classifier
from pygo.classifiers.CnnClassifier import CnnClassifier
from pygo.classifiers.MobilenetV4 import MobilenetV4Classifier
from pygo.classifiers.Convnext import ConvnextClassifier


from sklearn.ensemble import VotingClassifier
warnings.filterwarnings('always') 

@dataclass
class CV2PlotSettings:
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fontScale : float      = 0.8
    fontColor : Tuple[int] = (255,255,255)
    thickness : int        = 1
    lineType  : int        = 2

class EnsembleDebugKeys(Enum):
    Mask_Black = auto()
    Mask_White = auto()
    Detected_Intensities = auto()
    IMG_B = auto()
    IMG_W = auto()
    MASK = auto()
    GRID = auto()
    BIN = auto()
    CIRCLE = auto()
    #DETECT = auto()
    DETECT0 = auto()    # circle
    DETECT1 = auto()
    DETECT2 = auto()
    DETECT3 = auto()
    DETECT4 = auto()




class EnsembleClassifier(Classifier, DebugInfoProvider, Timing):
    def __init__(self, BOARD:GoBoard, size: int) -> None:
        Classifier.__init__(self)
        Timing.__init__(self)
        DebugInfoProvider.__init__(self)

        self.classifier = CnnClassifier("weights.pt")
        self.classifier_2 = ConvnextClassifier("convnext.pt")
        self.classifier_3 = MobilenetV4Classifier("mobilenetv4.pt")
        #self.classifier_3 = HOGSVMClassifier()
        self.ensemble = VotingClassifier(estimators=[
                            ('cnn0', self.classifier), 
                            ('cnn1', self.classifier_2), 
                            ('svm', self.classifier_3)], voting='hard')

        self.size = size 
        self.hasWeights = True
        self.BOARD=BOARD
        self.img_debug = None
        self.grid = None
        self.grid_img = None
        self.factor = None
        self.grid_plot = Plot()


        for key in EnsembleDebugKeys:        
            self.available_debug_info[key.name] = False

        self.cv_settings =  CV2PlotSettings()
        CoreSignals.subscribe(OnBoardGridSizeKnown, self.update_grid_size)



    def image_to_patches(self, img: B3CImage) -> List[B3CImage]:
        w = (np.mean(np.diff(self.BOARD.go_board_shifted.reshape(19,19,2)[:,:,0], axis=0))//2).astype(int)+2
        h = (np.mean(np.diff(self.BOARD.go_board_shifted.reshape(19,19,2)[:,:,1], axis=1))//2).astype(int)+2
        patches = []
        for (x,y) in self.BOARD.go_board_shifted.astype(int):
            patches.append(img[x-w:x+w, y-h:y+h])
        return patches

    def update_grid_size(self, args):
        grid = args[0].reshape(19,19,2)
        dx = np.mean(np.diff(grid[:,:,0].T))
        dy = np.mean(np.diff(grid[:,:,1]))
        a = ((np.mean([dx,dy])/2)**2)*np.pi
        
        self.grid = grid    # save grid coordinates
        self.grid_img = None # clear old grid image to force repainting
        self.factor = None
  

    def predict(self, img: B3CImage) -> GoBoardClassification:
        #patches = self.image_to_patches(img)
        #val = self.ensemble.predict(patches)
        cnn_pred          = self.classifier.predict_proba(self.image_to_patches(img))
        cnn2_pred         = self.classifier_2.predict_proba(self.image_to_patches(img))
        cnn3_pred         = self.classifier_3.predict_proba(self.image_to_patches(img))
        #hog_pred          = self.classifier_3.predict_prob(self.image_to_patches(img))
       
        val = cnn_pred + cnn2_pred + cnn3_pred
        val_tmp = np.argmax(val, axis=-1)
        val = np.zeros_like(val_tmp)
        
        # remap classes
        val[val_tmp==0] = 2
        val[val_tmp==1] = 0
        val[val_tmp==2] = 1

        return val