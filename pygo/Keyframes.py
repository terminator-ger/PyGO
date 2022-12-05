import pickle
import logging
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from sre_parse import State
from pygo.utils.color import C2N
from pygo.utils.typing import B3CImage, GoBoardClassification
from pygo.Signals import *

@dataclass
class Keyframe:
    #img         : B3CImage              = None
    timestamp   : float                 = None
    prediction  : GoBoardClassification = None
    state       : GoBoardClassification = None
    #move_added  : bool                  = None
    #colour      : int                   = None


class History:
    def __init__(self):
        self.keyframes = {}

    def get_kf(self, time: float) -> Keyframe:
        '''
        returns the closes keyframe to time
        '''
        t = min(self.keyframes.keys(), key=lambda x:abs(x-time))
        return self.keyframes[t]


    #def update(self, img: B3CImage, ts: float, pred: GoBoardClassification, new_move: bool, colour:int) -> None:
    def update(self, ts: float, pred: GoBoardClassification, state: GoBoardClassification) -> None:
        #self.keyframes[ts] = Keyframe(img, ts, pred, new_move, colour)
        self.keyframes[ts] = Keyframe(ts, pred, state)
        logging.debug('New Keyframe at {}'.format(ts))
        logging.debug('State: \n {}'.format(pred))
        #with open('history.pkl'.format(ts), 'wb') as f:
        #    pickle.dump(self.keyframes, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open('history.pkl', 'rb') as f:
            self.keyframes = pickle.load(f)
   
    def how_many_kf_since_last_update(self) -> Tuple[float, int]:
        ipt_ts = list(self.keyframes.keys())

        cnt_state = 0
        deviating_state = False
        for i in reversed(range(len(ipt_ts))):
            if i == 0:
                return (ipt_ts[-1], 0)   # stop at first element, we went all trought the history
            ts = ipt_ts[i]
            prev_ts = ipt_ts[i-1]
            prev = self.keyframes[prev_ts].state
            state = self.keyframes[ts].state
            pred = self.keyframes[ts].prediction

            diff_state = np.abs(prev-state)
            diff_idx_state = np.argwhere(diff_state)
            diff_state_sum = diff_idx_state.sum()
            diff_idx_pred = np.argwhere(np.abs(pred-state))
            cnt_pred = 0
            for j in range(len(diff_idx_pred)):
                x = diff_idx_pred[j][0]
                y = diff_idx_pred[j][0]
                if state[x,y] == C2N("E"):#in [C2N('B'), C2N('W')]:
                    cnt_pred += 1
            
            if cnt_pred > 0:
                deviating_state = True

            if not deviating_state:
                # last state was updated without problems
                logging.debug('No deviating history detected')
                return (ipt_ts[i], cnt_state)

            elif deviating_state and diff_state_sum == 0 and cnt_pred == 0:
                # we found the last kf where the state was updated
                logging.debug('Deviating history detected - {} kf have differences'.format(cnt_state))
                return (ipt_ts[i], cnt_state)

            if (diff_state_sum == 0 and cnt_pred > 0) or deviating_state:
                cnt_state += 1    # one more kf without update has passed
                cnt_pred  = 0


            #else we continue to search further back in time

