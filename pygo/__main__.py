import threading

from pygo.ui.pygotk import PyGOTk
from pygo.core import PyGO
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    return parser.parse_args()

def run_app_threaded(args):
    core = PyGO(args)
    core_thread = threading.Thread(target=core.loop)
    core_thread.start()

    ui = PyGOTk(pygo=core)
    ui.run()

    core_thread.join()

def run_app_singlecore(**args):
    ui = PyGOTk()
    ui.run()

run_app = run_app_threaded

if __name__ == '__main__':
    args = parse_args()
    #run_app_singlecore()
    run_app_threaded(args)