import logging

from pygo.Signals import Signals, OnSettingsChanged
from pygo.Game import MoveValidationAlg

PyGOSettings = {
    # Motion Detection 
    'MotionDetectionBoard' : 0.02,
    'MotionDetectionBorder' : 0.001,

    # Game
    'AllowUndo': False,
    'MoveValidation': MoveValidationAlg.TWO_MOVES
}


def settings_updated(args):
    new_settings = args[0]
    logging.info("Settings updated:")
    for k in PyGOSettings.keys():
        if k in new_settings.keys():
            PyGOSettings[k] = new_settings[k].get()
        logging.info("{} : {}".format(k, PyGOSettings[k]))


Signals.subscribe(OnSettingsChanged, settings_updated)