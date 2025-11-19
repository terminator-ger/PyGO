import logging
from enum import Enum, auto

from pygo.Signals import CoreSignals, OnSettingsChanged

class MoveValidationAlg(Enum):
    NONE = auto()
    ONE_MOVE = auto()
    TWO_MOVES = auto()
    MULTI_MOVES = auto()

class CORNER_DETECTION_ALG(Enum):
    WITH_VP = auto()
    FAST = auto()
    CPD = auto()


PyGOSettings = {
    # Motion Detection 
    'MotionDetectionBoard' : 0.02,
    'MotionDetectionBorder' : 0.001,
    
    # Board Detection
    'StaticBoard': True,
    'Export': True,
    'CornerDetectionAlg': CORNER_DETECTION_ALG.CPD,

    # Game
    'AllowUndo': False,
    'Backtrack': False,
    'MoveValidation': MoveValidationAlg.TWO_MOVES
}


