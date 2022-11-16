import numpy as np

from collections import deque
from typing import List, Tuple
from nptyping import NDArray, Int, Shape

from pygo.utils.typing import Corners


class MovingStatistics:
    def __init__(self, window=15):
        self.std  = None
        self.mean = None
        self.k = 0
    
    def update(self, x: Corners) -> Tuple[NDArray, NDArray]:
        if self.k == 0:
            self.mean = x
            self.std = np.zeros_like(x)
            self.k = 1
        else:
            mean_last = self.mean
            self.k += 1
            self.mean = mean_last + (x - mean_last)/self.k
            self.std = self.std + (x - mean_last)*(x - self.mean)
        return self.mean, self.std

    def reset(self):
        self.k = 0
        self.std = None
        self.mean = None