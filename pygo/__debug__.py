from pygo.core import PyGO
from pygo.ui.pygotk import PyGOTk
import threading
import logging


if __name__ == "__main__":
    #pygo = PyGO()
    #ui = PyGOTk(pygo)
    #t_l = threading.Thread(target=pygo.loop).start()
    #ui.run()
    #t_l.join()
    ui = PyGOTk()
    ui.run()

