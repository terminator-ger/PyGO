from pygo.core import PyGO
from pygo.pygotk import PyGOTk

import cProfile


if __name__ == "__main__":
    pygo = PyGO()
    #ui = PyGOTk(pygo=pygo)
    cProfile.run('pygo.loop10x()')
    #ui.run()
    #pygo.loop()

