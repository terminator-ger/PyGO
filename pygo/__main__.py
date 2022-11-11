from pygo.core import PyGO
from pygo.ui.pygotk import PyGOTk
import threading


def run_app():
    pygo = PyGO()
    ui = PyGOTk(pygo)
    t_l = threading.Thread(target=pygo.loop).start()
    ui.run()
    t_l.join()

if __name__ == "__main__":
    run_app()
