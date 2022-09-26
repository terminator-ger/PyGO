from optparse import Option
from typing import Optional
import cv2

from typing import Optional, List, Dict, Tuple
from enum import Enum
import pdb
from utils.misc import cv2Input, flattenList
import logging
from utils.typing import Image

class DebugInfoProvider:
    def __init__(self) -> None:
        self.available_debug_info : Dict[str, bool] = {}
        self.debugkeys : Optional[Enum] = None

    def getAvailableDebugOptions(self) -> Optional[List[str]]:
        return self.available_debug_info.keys()

    def enable(self, key: str) -> None:
        self.available_debug_info[key] = True
    
    def disable(self, key: str) -> None:
        self.available_debug_info[key] = False

    def showDebug(self, key: str, img: Image) -> None:
        if self.available_debug_info[key]:
            cv2.imshow(key, img)
            cv2.waitKey(1)
        else:
            try:
                cv2.destroyWindow(key)
            except Exception:
                pass




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

