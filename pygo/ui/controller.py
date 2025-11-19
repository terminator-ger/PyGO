from pygo.ui.pygotk import PyGOTk
from pygo.core import PyGO
from pygo.Signals import *

class Controller:
    def __init__(self, view: PyGOTk, core: PyGO):
        self.view = view
        self.core = core
        self.connect_signals()
    
    def connect_signals(self):
        # Connect UI signals to core methods
        #UISignals.subscribe(OnStartNewGame, self.core.startNewGame)
        #UISignals.subscribe(OnLoadHistory, self.core.History.load)
        #UISignals.subscribe(OnSaveHistory, self.core.History.save)
        #UISignals.subscribe(OnUndoMove, self.core.undoLastMove)
        #UISignals.subscribe(OnRedoMove, self.core.redoMove)
        #UISignals.subscribe(OnSetBoardSize, self.core.setBoardSize)
        #UISignals.subscribe(OnSetKomi, self.core.setKomi)
        #UISignals.subscribe(OnSetHandicapStones, self.core.setHandicapStones)
        #UISignals.subscribe(OnPlayMove, self.core.playMove)
        
        # Connect core signals to UI methods
        #CoreSignals.subscribe(OnGameStateUpdated, self.view.updateGameState)
        #CoreSignals.subscribe(OnBoardUpdated, self.view.updateBoard)
        #CoreSignals.subscribe(OnScoreUpdated, self.view.updateScore)
        #CoreSignals.subscribe(OnHistoryLoaded, self.view.refreshHistoryView)
        #CoreSignals.subscribe(OnHistorySaved, self.view.notifyHistorySaved)
        UISignals.subscribe(NewInputFile, self._NewInputFile)
    
    def parse_args(self, args):
        if args.video:
            self._NewInputFile([args.video])
    
    def _NewInputFile(self, args):
        input_data = args[0]
        self.core.input_stream.set_input_file_stream(input_data)
        CoreSignals.emit(GameReset, 19)
        if '/dev/video' in input_data:
            # camera
            self.view.hide_video_ui()
            self.view.go_tree_pause["state"] = "normal"
        else:
            self.view.show_video_ui()
            self.view.time_slider.reset()
            self.view.time_slider.on_update_time(self.core.input_stream.get_length())
            self.view.onGameNew()
            self.view.go_tree_pause["state"] = "disabled"