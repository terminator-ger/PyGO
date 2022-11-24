import pickle
import logging
from dataclasses import dataclass
from pygo.utils.typing import B3CImage, GoBoardClassification
from pygo.Signals import *

@dataclass
class Keyframe:
    #img         : B3CImage              = None
    timestamp   : float                 = None
    prediction  : GoBoardClassification = None
    #move_added  : bool                  = None
    #colour      : int                   = None


class History:
    def __init__(self):
        self.keyframes = {}


    #def update(self, img: B3CImage, ts: float, pred: GoBoardClassification, new_move: bool, colour:int) -> None:
    def update(self, ts: float, pred: GoBoardClassification) -> None:
        #self.keyframes[ts] = Keyframe(img, ts, pred, new_move, colour)
        self.keyframes[ts] = Keyframe(ts, pred)
        logging.debug('New Keyframe at {}'.format(ts))
        logging.debug('State: \n {}'.format(pred))
        #with open('history.pkl'.format(ts), 'wb') as f:
        #    pickle.dump(self.keyframes, f, protocol=pickle.HIGHEST_PROTOCOL)


   
    def how_many_kf_since_last_update(self, timestamp: float):
        ipt_ts = reversed(sorted(self.keyframes.keys()))
        cnt = 0
        for ts in ipt_ts:
            cnt += 1
            if self.keyframes[ts].move_added:
                break
        return cnt
