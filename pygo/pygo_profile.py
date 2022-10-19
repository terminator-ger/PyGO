from pygo.core import PyGO
from pygo.pygotk import PyGOTk

import cProfile


if __name__ == "__main__":
    pygo = PyGO()
    #ui = PyGOTk(pygo=pygo)
    #ui.run()
    cProfile.run('pygo.loopDetect10x()')
    #ui.quit()
    #ui.run()
    #pygo.loop()

