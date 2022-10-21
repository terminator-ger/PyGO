from enum import Enum

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

