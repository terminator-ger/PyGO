from pygo.core import PyGO

import cProfile
import pstats

if __name__ == "__main__":
    pygo = PyGO()
    #ui = PyGOTk(pygo=pygo)
    #ui.run()

    with cProfile.Profile() as pr:
        pygo.loop10x()
        pr.create_stats()
        pr.dump_stats('profile_view.stats')
    #cProfile.run('pygo.loopDetect10x()')
    #ui.quit()
    #ui.run()
    #pygo.loop()

