from abc import ABC, abstractmethod
from queue import Queue
from typing import Callable
import logging

class Signals(ABC):
    pass

class UISignals(Signals):
    subs = {}
    emittedQ = Queue()

    @staticmethod
    def subscribe(sig, fun : Callable):
        logging.debug("Function {} was registered for {}".format(fun , sig))
        sig = sig.__name__
        if sig not in UISignals.subs.keys():
            UISignals.subs[sig] = []
        UISignals.subs[sig].append(fun)

    @staticmethod
    def emit(sig, *args):
        sig = sig.__name__
        if len(args) == 0:
            args = None
        logging.debug("UISignals emit: {}".format(sig))
        if sig in UISignals.subs.keys():
            UISignals.emittedQ.put((sig, args))

    @staticmethod
    def process_signals():
        while not UISignals.emittedQ.empty():
            sig_emitted = UISignals.emittedQ.get()
            sig = sig_emitted[0]
            args = sig_emitted[1]

            for sub in UISignals.subs[sig]:
                sub(args)

class CoreSignals(Signals):
    subs = {}
    emittedQ = Queue()
    @staticmethod
    def subscribe(sig, fun : Callable):
        logging.debug("Function {} was registered for {}".format(fun , sig))
        sig = sig.__name__
        if sig not in CoreSignals.subs.keys():
            CoreSignals.subs[sig] = []
        CoreSignals.subs[sig].append(fun)

    @staticmethod
    def emit(sig, *args):
        sig = sig.__name__
        if len(args) == 0:
            args = None
        logging.debug("CoreSignals emit: {}".format(sig))
        if sig in CoreSignals.subs.keys():
            CoreSignals.emittedQ.put((sig, args))

    @staticmethod
    def process_signals():
        while not CoreSignals.emittedQ.empty():
            sig_emitted = CoreSignals.emittedQ.get()
            sig = sig_emitted[0]
            args = sig_emitted[1]

            for sub in CoreSignals.subs[sig]:
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

class UpdateHistory(Signals):
    pass

class NewMove(Signals):
    pass


# Video Navigation
class VideoFrameCounterUpdated(Signals):
    pass

class InputStreamSeek(Signals):
    pass

class InputBackward(Signals):
    pass

class InputBackward10(Signals):
    pass

class InputForward(Signals):
    pass

class InputForward10(Signals):
    pass

class PreviewNextFrame(Signals):
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

class Exit(Signals):
    pass



class UIDrawStoneOnTimeline(Signals):
    pass

class UIUpdateLog(Signals):
    pass

class UIOnBoardDetected(Signals):
    pass

class UIOnBoardReset(Signals):
    pass

class UIGameReset(Signals):
    pass

class UIVideoFrameCounterUpdated(Signals):
    pass