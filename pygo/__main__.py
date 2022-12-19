import threading

from pygo.ui.pygotk import PyGOTk
from pygo.core import PyGO

def run_app_threaded():
    core = PyGO()
    core_thread = threading.Thread(target=core.loop)
    core_thread.start()

    ui = PyGOTk(pygo=core)
    ui.run()

def run_app_singlecore():
    ui = PyGOTk()
    ui.run()


if __name__ == '__main__':
    run_app_singlecore()
