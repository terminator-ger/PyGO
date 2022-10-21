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



class OnBoardDetected(Signals):
    pass

class OnBoardGridSizeKnown(Signals):
    pass

class OnSettingsChanged(Signals):
    pass

class OnBoardMoved(Signals):
    pass