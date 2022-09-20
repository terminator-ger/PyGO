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

GoBoardClassification = NDArray[Shape["19,19"], Int]



