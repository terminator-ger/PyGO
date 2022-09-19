
class CameraCalib:
    def __init__(self, intr) -> None:
        self.focal = intr[0,0]
        self.intr = intr
        self.center = (intr[0,2], intr[1,2])

