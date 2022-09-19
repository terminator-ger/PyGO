from enum import Enum

def N2C(c):
    if c == 0:
        c_str = 'W' 
    elif c == 1:
        c_str = 'B' 
    elif c == 2:
        c_str = 'E' 
    return c_str
def C2N(c):
    if c == 'W':
        return 0
    elif c == 'B':
        return 1
    elif c == 'E':
        return 2

class Color(Enum):
    WHITE = 0
    BLACK = 1
    NONE  = 2

