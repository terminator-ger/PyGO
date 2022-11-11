#from pygo.core import PyGO
from pygo.ui.pygotk import PyGOTk
#import threading


def run_app():
    ui = PyGOTk()
    ui.run()
    #pygo = PyGO()
    #t_l = threading.Thread(target=pygo.loop).start()
    #t_l.join()

if __name__ == "__main__":
    run_app()
