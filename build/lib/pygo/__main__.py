from pygo.pygo import PyGOTk, PyGO



if __name__ == "__main__":
    #pygo = PyGO()
    #ui = TkPyGO(pygo)
    #t_l = threading.Thread(target=pygo.loop).start()
    #ui.run()
    #t_l.join()
    ui = PyGOTk()
    ui.run()

