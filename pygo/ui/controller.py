from pygo.ui.pygotk import PyGOTk
from pygo.core import PyGO
from pygo.Signals import *

class Controller:
    def __init__(self, view: PyGOTk, core: PyGO):
        self.view = view
        self.core = core
        self.connect_signals()
    
    def connect_signals(self):
        self.view.connect_signals()
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