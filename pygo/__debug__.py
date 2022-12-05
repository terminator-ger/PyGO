from pygo.Signals import Signals
from pygo.core import PyGO
from pygo.ui.pygotk import PyGOTk
from pygo.Signals import *
from pygo.utils.color import C2N
import pdb


if __name__ == "__main__":
    ui = PyGOTk()
    # load history and video
    ui.load_video('/home/michael/dev/PyGo/go-spiel.mp4')
    ui.pygo.startNewGame()

    T = 194.0
    ui.pygo.History.load()
    ui.pygo.input_stream.set_pos((T,)) # 3:14
    # restore last state
    state = ui.pygo.History.get_kf(T).state
    ui.pygo.Game.state = state
    ui.pygo.Game.last_x = 0
    ui.pygo.Game.last_y = 0
    ui.pygo.Game.last_color = C2N('W')
    ui.run()

