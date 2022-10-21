from time import sleep

from pip import main
import cv2
import numpy as np
import pdb
import threading
from datetime import datetime
import PIL
from PIL import ImageTk
import logging
from functools import partial


import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import tkinter.scrolledtext as scrolledtext


from pygo.Signals import GamePauseResume, GameTreeBack, GameTreeForward, Signals
from pygo.core import PyGO
from pygo.classifier import GoClassifier, HaarClassifier, IlluminanceClassifier, CircleClassifier
from pygo.Motiondetection import MotionDetectionMOG2
from pygo.GoBoard import GoBoard
from pygo.utils.typing import B3CImage, Image, NetMove
from pygo.utils.data import save_training_data
from pygo.utils.misc import flattenList
from pygo.utils.debug import DebugInfo
from pygo.Game import Game, GameState
from pygo.Ensemble import SoftVoting, MajorityVoting
from pygo.Webcam import Webcam
from pygo.Signals import Signals, OnSettingsChanged

class PyGOTk:
    #types
    img_cam: Image
    img_overlay: Image

    def __init__(self, pygo: PyGO = None):
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)
        if pygo is None:
            self.pygo = PyGO()
            self.weOwnControllLooop = True
        else:
            self.pygo = pygo 
            self.weOwnControllLooop = False

        self.DebugInfo = DebugInfo([self.pygo.Motiondetection, 
                                    #self.pygo.BoardMotionDetecion, 
                                    self.pygo.Board, 
                                    self.pygo.Game,
                                    self.pygo.PatchClassifier])


        self.grid = None
        self.lock_grid = threading.Lock()


        self.root = tk.Tk()
        self.root.title('PyGO')
        
        self.menubar = tk.Menu(self.root, tearoff=0)

        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="Save", command=self.onFileSave)
        filemenu.add_command(label="Settings", command=self.onSettings)
        filemenu.add_command(label="Exit", command=self.onFileExit)

        boardmenu = tk.Menu(self.menubar, tearoff=0)
        boardmenu.add_command(label='Detect', command=self.onBoardDetect)

        gamemenu = tk.Menu(self.menubar, tearoff=0)
        gamemenu.add_command(label='Start new', command=self.onGameNew)
        
        self.viewVar = tk.IntVar(value=0)
        viewmenu = tk.Menu(self.menubar, tearoff=0)
        viewmenu.add_radiobutton(label="Overlay",  value=0, variable=self.viewVar)
        viewmenu.add_radiobutton(label="Original", value=1, variable=self.viewVar)
        viewmenu.add_radiobutton(label="Virtual",  value=2, variable=self.viewVar)


        self.menubar.add_cascade(label="File", menu=filemenu)
        self.menubar.add_cascade(label="Game", menu=gamemenu)
        self.menubar.add_cascade(label="Board", menu=boardmenu)
        self.menubar.add_cascade(label="View", menu=viewmenu)

        self.root.config(menu=self.menubar)
        
        debugmenu = tk.Menu(self.menubar, tearoff=0)
        self.img_overlay = self.pygo.img_overlay

        debuglevelmenu = tk.Menu(debugmenu, tearoff=0)
        debuglevelmenu.add_checkbutton(label='Info', command=self.setLogLevelInfo)
        debuglevelmenu.add_checkbutton(label='Debug', command=self.setLogLevelDebug)
        debuglevelmenu.add_checkbutton(label='Warn', command=self.setLogLevelWarn)

        debugviewsmenu = tk.Menu(debugmenu, tearoff=0)
        fn, optn = self.DebugInfo.getOptions()
        self.fn = fn
        self.optn = optn
        self.DebugViewStates = [tk.BooleanVar(False) for _ in fn]

        for f,o,s in zip(fn, optn, self.DebugViewStates):
            debugviewsmenu.add_checkbutton(label=o, 
                                            variable=s,
                                            command=partial(self.switchState, f, o, s))
        

        debugmenu.add_cascade(label='Loglevel', menu=debuglevelmenu)
        debugmenu.add_cascade(label='Views', menu=debugviewsmenu)
        self.menubar.add_cascade(label="Debug", menu=debugmenu)


        self.tkimage = self.__np2tk(self.pygo.img_cam)
        self.go_board_display = tk.Label(image=self.tkimage)
        self.go_board_display.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.go_board_display.image = self.tkimage

        self.move_log = scrolledtext.ScrolledText(self.root, undo=True, width=10)
        self.move_log.grid(column=1, row=0, padx=5, pady=5)

        self.sep_h = ttk.Separator(self.root, orient='horizontal')
        self.sep_h.grid(column=0, row=1, sticky='ew')

        self.go_tree_display = tk.PanedWindow(self.root)
        self.go_tree_display.grid(column=0, row=2)

        self.go_tree_bwd   = tk.Button(self.go_tree_display, text="<=", command=self.GameTreeBack)
        self.go_tree_bwd.grid(column=0, row=0)
        self.go_tree_pause = tk.Button(self.go_tree_display, text="|>", command=self.GamePause)
        self.go_tree_pause.grid(column=1, row=0)
        self.go_tree_fwd   = tk.Button(self.go_tree_display, text="=>", command=self.GameTreeForward)
        self.go_tree_fwd.grid(column=2, row=0)


        self._next_job = None
        self.QUIT = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.settings = {'AllowUndo' : tk.BooleanVar(value=False),
                         'MotionDetectionFactor': tk.DoubleVar(value=0.9),
        }

    def GamePause(self) -> None:
        Signals.emit(GamePauseResume)
   
    def GameTreeBack(self) -> None:
        Signals.emit(GameTreeBack)

    def GameTreeForward(self) -> None:
        Signals.emit(GameTreeForward)
    
    def switchState(self, fn, name, state):
        if state.get():
            fn.enable(name)
        else:
            fn.disable(name)

    def setLogLevelInfo(self) -> None:
        logging.getLogger().setLevel(logging.INFO)

    def setLogLevelDebug(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)

    def setLogLevelWarn(self) -> None:
        logging.getLogger().setLevel(logging.WARN)

    def onSettings(self):
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title('Settings')
        self.settings_window.grid()
        packlist = []
        btn1 = tk.Checkbutton(self.settings_window, 
                                text='Allow undoing moves during recording',
                                variable=self.settings['AllowUndo'],
                                onvalue=True, 
                                offvalue=False)
        packlist.append(btn1)

        lbl1 = tk.Label(self.settings_window, text="Motion Detection Agressiveness")
        packlist.append(lbl1)

        sep = ttk.Separator(self.settings_window, orient='horizontal')
        packlist.append(sep)

        switch_frame = tk.Frame(self.settings_window)
        packlist.append(switch_frame)

        low_button = tk.Radiobutton(switch_frame, 
                                    text="Low", 
                                    variable=self.settings['MotionDetectionFactor'],
                                    indicatoron=False, 
                                    value=0.9, 
                                    width=8)
        med_button = tk.Radiobutton(switch_frame, 
                                    text="Medium", 
                                    variable=self.settings['MotionDetectionFactor'],
                                    indicatoron=False, 
                                    value=0.92, 
                                    width=8)
        high_button = tk.Radiobutton(switch_frame, 
                                    text="High", 
                                    variable=self.settings['MotionDetectionFactor'],
                                    indicatoron=False, 
                                    value=0.95, 
                                    width=8)
        low_button.pack(side="left")
        med_button.pack(side="left")
        high_button.pack(side="left")
        for item in packlist:
            item.pack()

        self.settings_window.protocol("WM_DELETE_WINDOW", self.on_settings_closing)

    def on_settings_closing(self):
        logging.debug("Settings changed")
        Signals.emit(OnSettingsChanged, self.settings)

        self.settings_window.destroy()


    def on_closing(self):
        if self.pygo.Game.game_tree is not None and len(self.pygo.Game.game_tree.get_root()) > 0:
            if tk.messagebox.askokcancel("Quit", "Do you want to quit without saving?"):
                self.QUIT = True
                self.root.destroy()
        else:
            self.QUIT = True
            self.root.destroy()

    def quit(self):
        self.pygo.webcam.release()
        self.root.quit()
        self.root.destroy()


    def onBoardDetect(self) -> None:
        self.pygo.Board.calib(self.pygo.img_cam)
        self.updateGrid()

    def onFileOpen(self) -> None:
        return

    def onFileSave(self) -> None:
        cur_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
        with fd.asksaveasfile(mode='wb', 
                                initialfile='Game_{}.sgf'.format(cur_time),
                                defaultextension='sgf',
                                filetypes=[("Smart Game Format",".sgf")]) as file:
            self.pygo.Game.saveGame(file)

    def onFileExit(self) -> None:
        self.on_closing()

    def onGameNew(self) -> None:
        cur_time = datetime.now().strftime("%d-%m-%Y")
        self.move_log.delete('1.0', tk.END)
        self.move_log.insert('end', 'New Game\n')
        self.move_log.insert('end', '{}\n'.format(cur_time))
        self.move_log.insert('end', '==========\n')
        self.pygo.startNewGame()

    def run(self) -> None:
        self.root.after(1, self.update)
        self.root.mainloop()

    def logMove(self, msg: NetMove) -> None:
        if msg is not None:
            if msg[0] == 'e':
                self.move_log.delete('end-2l', 'end-1l')
            elif msg[0] == '':
                pass
            else:
                self.move_log.insert('end', '{}: {}-{}\n'.format(msg[0], msg[1]+1, msg[2]+1))
            self.move_log.see('end')  # move to the end after adding new text


        #if not self.QUIT:
        #    self._next_job = self.root.after(1, self.loop)
        #else:
        #    self.quit()


    def __np2tk(self, img : Image) -> ImageTk.PhotoImage: 
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(PIL.Image.fromarray(rgb))

    def updateGrid(self) -> None:
        with self.lock_grid:
            self.grid = self.pygo.Board.go_board_shifted


    def update(self) -> None:
        if self.weOwnControllLooop:
            self.pygo.run_once()
        
        #state = self.pygo.Game.getCurrentState()
        ##self.img_overlay = self.pygo.img_overlay.copy()
        ##self.img_overlay = self.pygo.img_cam.copy()
        #kif (state is not None and \
        #        self.grid is not None and\
        #        self.pygo.Board.hasEstimate):
            #cv2.imwrite('out.png', self.img_overlay)
            #self.img_overlay = plot_overlay(state, self.grid, self.img_overlay)

        if str(self.pygo.msg) != '':
            self.logMove(self.pygo.msg)


        if self.pygo.Game.GS == GameState.RUNNING:
            self.go_tree_pause.configure(text='||')
        elif self.pygo.Game.GS == GameState.PAUSED:
            self.go_tree_pause.configure(text='|>')
    

        # switch view
        view = self.viewVar.get()
        if view == 0:
            self.tkimage = self.__np2tk(self.pygo.img_overlay)
        elif view == 1:
            self.tkimage = self.__np2tk(self.pygo.img_cropped)
        elif view == 2:
            self.tkimage = self.__np2tk(self.pygo.img_virtual)

        self.go_board_display.configure(image=self.tkimage)
        self.go_board_display.image = self.tkimage
        self.root.after(1, self.update)


    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x_grid = np.repeat(x, 19*19)
            y_grid = np.repeat(y, 19*19)
            ref = np.stack((x_grid, y_grid)).T
            dist = np.mean((ref - self.grid)**2, axis=1)
            coord = np.argmin(dist)
            x_board, y_board = np.unravel_index(coord, (19,19))
            self.pygo.Game.setManual(x_board, y_board)
            #self.update()

           # draw circle here (etc...)


