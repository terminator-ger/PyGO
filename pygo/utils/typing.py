from typing import Tuple, List, Optional, Union
import numpy as np
from nptyping import NDArray, Int, Shape
from nptyping import UInt8, Float32, Float

Move = Tuple[str, Tuple[int, int]]
NetMove = List[Union[str, int]] # in serial format to be sent over socket

Point2D = Tuple[Float, Float]
Point3D = Tuple[Float, Float, Float]
Corners = List[Point2D]

B1CImage = NDArray[Shape["*,*,1"], UInt8]
B3CImage = NDArray[Shape["*,*,3"], UInt8]
F1CImage = NDArray[Shape["*,*,1"], Float32]
F3CImage = NDArray[Shape["*,*,3"], Float32]

Image = NDArray
Patches = List[Image]
Mask = B1CImage

Homography = NDArray[Shape["3,3"], Float32]

FlattenGoBoardC19 = NDArray[Shape["1,361"], Int]
FlattenGoBoardC13 = NDArray[Shape["1,169"], Int]
FlattenGoBoardC9 = NDArray[Shape["1,81"], Int]
FlattenGoBoard = Union[FlattenGoBoardC19, FlattenGoBoardC13, FlattenGoBoardC9]

GoBoardClassification19 = NDArray[Shape["19,19"], Int]
GoBoardClassification13 = NDArray[Shape["13,13"], Int]
GoBoardClassification9 = NDArray[Shape["9,9"], Int]
GoBoardClassification = Union[GoBoardClassification19, GoBoardClassification13, GoBoardClassification9]



Line = NDArray[Shape["4,1"], Float32]
Lines = NDArray[Shape["*,4"], Float32]
