from abc import ABC, abstractmethod
from typing import Callable

class Signals(ABC):
    subs = {}

    @staticmethod
    def subscribe(sig, fun : Callable):
        sig = sig.__name__
        if sig not in Signals.subs.keys():
            Signals.subs[sig] = []
        Signals.subs[sig].append(fun)

    @staticmethod
    def emit(sig, *args):
        sig = sig.__name__
        if sig in Signals.subs.keys():
            for sub in Signals.subs[sig]:
                sub(args)


class DetectBoard(Signals):
    pass

class GameNew(Signals):
    pass

class OnBoardDetected(Signals):
    pass

class OnBoardGridSizeKnown(Signals):
    pass

class OnSettingsChanged(Signals):
    pass

class OnBoardMoved(Signals):
    pass

class GameTreeBack(Signals):
    pass

class GameTreeForward(Signals):
    pass

class GamePauseResume(Signals):
    pass

class OnCameraGeometryChanged(Signals):
    pass

class OnInputChanged(Signals):
    pass

class OnGridSizeUpdated(Signals):
    pass

class GameNewMove(Signals):
    pass