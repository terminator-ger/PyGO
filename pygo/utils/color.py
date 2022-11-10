from enum import Enum
from typing import Union
import numpy as np

def N2C(n: int) -> str:
    if n == 0:
        c_str = 'W' 
    elif n == 1:
        c_str = 'B' 
    elif n == 2:
        c_str = 'E' 
    return c_str


def C2N(c: str) -> int:
    if c.upper() == 'W':
        return 0
    elif c.upper() == 'B':
        return 1
    elif c.upper() == 'E':
        return 2
    elif c is None:
        return 2


class COLOR(Enum):
    WHITE = 0
    BLACK = 1
    NONE  = 2

def CNOT(c: Union[str,int]) -> Union[str,int]:
    if isinstance(c, int) or isinstance(c, np.int64):
        return CNOT_INT(c)
    else:
        return N2C(CNOT_INT(c))

def CNOT_INT(n: int):
    if n == 0:
        return 1
    elif n == 1:
        return 0
    else:
        return 2