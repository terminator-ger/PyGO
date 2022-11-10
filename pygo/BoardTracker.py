import numpy as np
import logging
from typing import Optional

from pygo.utils.debug import DebugInfoProvider
from pygo.utils.typing import B3CImage, Corners


class BoardTracker(DebugInfoProvider):
    def __init__(self):
        DebugInfoProvider.__init__(self)
        self.ref = None
        self.S = None
        self.std = 1

        self.s_h = 0
        self.s_l = 0
        self.w = 0
        self.T = 1
    
    def reset(self, corners:Corners) -> None:
        self.ref = corners
        self.s_h = 0
        self.s_l = 0

    def update(self, corners: Optional[Corners]) -> bool:
        if self.ref is None and corners is not None:
            self.ref = corners
            return False
        else:
            if corners is not None:
                # to zero mean abs distances
                z_n = (corners - self.ref ) 
                # euclidean distance
                z_n = np.sqrt(np.sum(z_n ** 2, axis=1)) / self.std 
                self.s_h = np.maximum(0, self.s_h + z_n - self.w)
                self.s_l = np.maximum(0, self.s_h - z_n - self.w)

                logging.debug2("s_h: {}".format(self.s_h))
                logging.debug2("s_l: {}".format(self.s_l))

                if np.any(self.s_h > self.T) or np.any(self.s_l > self.T):
                    logging.warning('Board was moved!')
                    self.reset(corners)
                    return True
                else:
                    return False
            else:
                return False



