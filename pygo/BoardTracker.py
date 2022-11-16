import numpy as np
import logging
from typing import Optional

from pygo.utils.debug import DebugInfoProvider
from pygo.utils.typing import B3CImage, Corners
from pygo.utils.movingstatistics import MovingStatistics


class BoardTracker(DebugInfoProvider):
    def __init__(self):
        DebugInfoProvider.__init__(self)
        self.S = None
        self.std = 1

        self.s_h = 0
        self.w = 0
        self.T = 5
        self.window_corner_detection = 15
        self.corners_stats = MovingStatistics(window=self.window_corner_detection)
    
    def reset(self) -> None:
        self.s_h = 0
        self.corners_stats.reset()

    def update(self, corners: Optional[Corners]) -> bool:
        if corners is not None and self.corners_stats.mean is None:
            self.corners_stats.update(corners)
            return False
        else:
            if corners is not None:
                # to zero mean abs distances
                if self.corners_stats.mean is None:
                    z_n = corners
                else:
                    z_n = (corners - self.corners_stats.mean)
                std = 1

                # euclidean distance
                z_n = np.sqrt(np.sum(z_n ** 2, axis=1)) / std 
                self.s_h = np.maximum(0, self.s_h + z_n - self.w)

                logging.debug2("s_h: {}".format(self.s_h))

                shifted_corners_h = np.sum(self.s_h > self.T)
                _,_ = self.corners_stats.update(corners)

                if shifted_corners_h > 1:
                    logging.warning('Board was moved!')
                    self.reset()
                    return True
                else:
                    return False
            else:
                return False



