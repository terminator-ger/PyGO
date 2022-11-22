
import logging
from dataclasses import dataclass
from pygo.utils.typing import B3CImage, GoBoardClassification
from pygo.Signals import *

@dataclass
class Keyframe:
    img         : B3CImage              = None
    timestamp   : int                   = None
    prediction  : GoBoardClassification = None
    move_added  : bool                  = None
    colour       : int                   = None

class History:
    def __init__(self):
        self.keyframes = {}


    def update(self, img: B3CImage, ts: int, pred: GoBoardClassification, new_move: bool, colour:int) -> None:
        self.keyframes[ts] = Keyframe(img, ts, pred, new_move, colour)
        logging.debug('New Keyframe at {}'.format(ts))
        logging.debug('State: \n {}'.format(pred))
    
   
    def how_many_kf_since_last_update(self, timestamp: int):
        ipt_ts = reversed(sorted(self.keyframes.keys()))
        cnt = 0
        for ts in ipt_ts:
            cnt += 1
            if self.keyframes[ts].move_added:
                break
        return cnt
