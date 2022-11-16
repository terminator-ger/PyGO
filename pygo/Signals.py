from abc import ABC, abstractmethod
from typing import Callable
import logging

class Signals(ABC):
    subs = {}

    @staticmethod
    def subscribe(sig, fun : Callable):
        logging.debug("Function {} was registered for {}".format(fun , sig))
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


# Commands to different submodules/ Detected Events
class DetectBoard(Signals):
    pass

class DetectHandicap(Signals):
    pass
 
class OnBoardDetected(Signals):
    pass

class OnBoardGridSizeKnown(Signals):
    pass

class OnSettingsChanged(Signals):
    pass

class OnBoardMoved(Signals):
    pass

class OnCameraGeometryChanged(Signals):
    pass

class OnInputChanged(Signals):
    pass

class OnGridSizeUpdated(Signals):
    pass

class UpdateLog(Signals):
    pass

class NewMove(Signals):
    pass

class VideoFrameCounterUpdated(Signals):
    pass

# Game tree navigation and playing moves
class GameTreeBack(Signals):
    pass

class GameTreeForward(Signals):
    pass

class GameNew(Signals):
    pass

class GameNewMove(Signals):
    pass

class GamePause(Signals):
    pass

class GameRun(Signals):
    pass

class GameHandicapMove(Signals):
    pass

class GameReset(Signals):
    pass
