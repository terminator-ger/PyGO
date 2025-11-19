import cv2
import pdb
import logging
import numpy as np
import PIL
from typing import Optional, List, Dict, Tuple
from enum import Enum, auto

from pygo.utils.misc import cv2Input, flattenList
from pygo.utils.typing import Image
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image

class debugkeys(Enum):
    Detected_Lines = auto()
    Detected_Grid = auto()
    Affine_Registration = auto()
    Board_Outline = auto()


class DebugInfoProvider:
    root = None
    def __init__(self) -> None:
        self.available_debug_info : Dict[str, bool] = {}
        self.debugkeys : Optional[Enum] = None
        self.windows = {}
        
    @staticmethod
    def setTKRoot(tkroot):
        DebugInfoProvider.root = tkroot

    def getAvailableDebugOptions(self) -> Optional[List[str]]:
        return self.available_debug_info.keys()

    def enable(self, key: str) -> None:
        self.available_debug_info[key] = True
    
    def disable(self, key: str) -> None:
        self.available_debug_info[key] = False

    def debugStatus(self, key: str) -> bool:
        return self.available_debug_info[key.name]

    def showDebug(self, key: Enum, img):
        k = key.name

        # If debug disabled → close & remove window
        if not self.available_debug_info.get(k, False):
            if k in self.windows:
                self.windows[k].destroy()
                del self.windows[k]
            return

        # Create window once
        if k not in self.windows:
            win = tk.Toplevel(DebugInfoProvider.root)
            win.title(k)
            self.windows[k] = {
                "window": win,
                "label": None,
                "image_ref": None
            }

        win = self.windows[k]["window"]

        # Convert image BGR → RGB → Tk Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(img))

        # Keep reference so GC doesn't delete it
        self.windows[k]["image_ref"] = tk_img

        # Create label once
        if self.windows[k]["label"] is None:
            lbl = tk.Label(win, image=tk_img)
            lbl.pack()  # pack only once
            self.windows[k]["label"] = lbl
        else:
            # Update existing label image
            self.windows[k]["label"].configure(image=tk_img)



class DebugInfo:
    def __init__(self, modules: List) -> None:

        self.debugkeys : Enum = None
        self.Modules = modules

        #self.debug_hooks, self.module_lookup = 
        self.debug_hooks = flattenList([x.getAvailableDebugOptions() for x in modules])
        self.module_lookup = []
        for x in modules:
            for _ in range(len(x.getAvailableDebugOptions())):
                self.module_lookup.append(x)

    def getOptions(self) -> Tuple[List]:
        return self.module_lookup, self.debug_hooks

    def showOptions(self) -> None:
        print('Debug Options')
        for i, optn in enumerate(self.debug_hooks):
            print('({}) : {}'.format(i, optn))
        selection = cv2Input()

        if selection is not None and int(selection) < len(self.debug_hooks):
            selection = int(selection)
            optn = self.debug_hooks[selection]
            self.module_lookup[selection].enable(optn)
            print('Enabled {}'.format(self.module_lookup[selection]))

class Timing:
    def __init__(self):
        self.times = {}
        self.start = {}
        self.stop = {}
    
    def tic(self, name=''):
        if name not in self.times.keys():
            self.times[name] = []
        self.start[name]  = time.time()

    def toc(self, name=''):
        self.stop[name] = time.time()
        self.times[name].append(self.stop[name]-self.start[name])
        logging.info("{}: {}".format(name, self.running_mean(name)))

    def running_mean(self, name):
        x = self.times[name]
        N = len(x)
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)