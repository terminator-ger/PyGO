import logging
from enum import Enum, auto

from pygo.Signals import CoreSignals, OnSettingsChanged


class MoveValidationAlg(Enum):
    NONE = auto()
    ONE_MOVE = auto()
    TWO_MOVES = auto()
    MULTI_MOVES = auto()


PyGOSettings = {
    # Motion Detection 
    'MotionDetectionBoard' : 0.02,
    'MotionDetectionBorder' : 0.001,

    # Game
    'AllowUndo': False,
    'MoveValidation': MoveValidationAlg.TWO_MOVES
}


