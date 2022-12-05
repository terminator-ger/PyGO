import os
import cv2
import pdb
import numpy as np
import threading
import PIL
import logging

from PIL import ImageTk
from datetime import datetime
from functools import partial

import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as scrolledtext
from tkinter import filedialog as fd

from pygo.core import PyGO
from pygo.utils.typing import B3CImage, Image, NetMove
from pygo.utils.debug import DebugInfo
from pygo.utils.color import C2N
from pygo.Game import  GameState
from pygo.Signals import *
from pygo.ui.TimeSlider import TimeSlider

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
        gamemenu.add_command(label='New Game', command=self.onGameNew)
        gamemenu.add_separator()
        gamemenu.add_command(label='Pause', command=self.GamePause)
        gamemenu.add_command(label='Resume', command=self.GameRun)
        gamemenu.add_separator()
        gamemenu.add_command(label='Detect Handicap', command=self.onDetectHandicap)
        gamemenu.add_command(label='Clear Manual Stones', command=self.onClearManualAll)
        
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
        debuglevelmenu.add_checkbutton(label='Debug2', command=self.setLogLevelDebug2)
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
        debugmenu.add_command(label='Take Screenshot', command=self.save_image)
        self.menubar.add_cascade(label="Debug", menu=debugmenu)

        self.pane_left = tk.Frame(master=self.root)
        self.pane_right = tk.LabelFrame(master=self.root, text='Move Log')
        self.pane_left.grid(column=0, row=0)
        self.pane_right.grid(column=1, row=0, sticky=tk.NS)

        ''' Image display '''
        self.tkimage = self.__np2tk(self.pygo.img_cam)
        self.go_board_display = tk.Label(self.pane_left, image=self.tkimage)
        self.go_board_display.image = self.tkimage
        self.go_board_display.grid(column=0, row=0, padx=5, pady=5)


        ''' Move Log '''
        self.move_log = scrolledtext.ScrolledText(self.pane_right, undo=True, width=15)
        self.move_log.pack(expand=True, fill=tk.Y, side=tk.TOP)

        ''' Tree Navigation Tools '''
        self.go_tree_display = tk.LabelFrame(self.pane_left, text='Tree Navigation')
        self.go_tree_display.grid(column=0, row=1, sticky=tk.W+tk.EW, padx=2)
        self.go_tree_display.columnconfigure(0, weight=1)
        self.go_tree_display.columnconfigure(1, weight=1)
        self.go_tree_display.columnconfigure(2, weight=1)
        self.go_tree_bwd   = tk.Button(self.go_tree_display, text="<-", command=self.GameTreeBack)
        self.go_tree_pause = tk.Button(self.go_tree_display, text="|>", command=self.GameTogglePauseResume)
        self.go_tree_fwd   = tk.Button(self.go_tree_display, text="->", command=self.GameTreeForward)

        self.go_tree_bwd.grid(row=1, column=0, sticky=tk.W+tk.E, padx=5, pady=5)
        self.go_tree_pause.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        self.go_tree_fwd.grid(row=1, column=2, sticky=tk.W+tk.E, padx=5, pady=5)


        ''' History and Video Tools '''

        self.time_slider_box = tk.LabelFrame(self.pane_left, text='Timeline')
        self.time_slider_box.grid(column=0, row=3, sticky=tk.W+tk.E, padx=2)
        self.time_slider_box.columnconfigure(0, weight=1)
        self.time_slider = TimeSlider(self.time_slider_box, self.pygo)
        self.time_slider.grid(row=0, column=0, sticky=tk.W+tk.E, padx=5)
        self.hide_video_ui()

        self._next_job = None
        self.QUIT = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.settings = {'AllowUndo' : tk.BooleanVar(value=False),
                         'MotionDetectionBoard': tk.DoubleVar(value=0.6),
                         'MotionDetectionBorder': tk.DoubleVar(value=0.2)
        }

        self.contextMenu = tk.Menu(self.root, tearoff=False)
        self.contextMenu.add_command(label='White', command=self.addManualWhite)
        self.contextMenu.add_command(label='Black', command=self.addManualBlack)
        self.contextMenu.add_command(label='None', command=self.addManualNone)
        self.contextMenu.add_separator()
        self.contextMenu.add_command(label='Clear', command=self.removeManual)

        self.root.bind("<space>", self.GameTogglePauseResume)
        self.go_board_display.bind("<ButtonPress-1>", self.leftMouseOnGOBoard)
        self.go_board_display.bind("<ButtonPress-3>", self.rightMouseOnGOBoard)

        self.moveHistory = []

        Signals.subscribe(UpdateLog, self.updateLog)
        Signals.subscribe(OnBoardDetected, self.updateGrid)
        Signals.subscribe(GameReset, self.__clear_log)
        Signals.subscribe(UIDrawStoneOnTimeline, self.videoAddNewMove)
        Signals.subscribe(VideoFrameCounterUpdated, self.video_frame_counter_udpated)


    def video_frame_counter_udpated(self, args):
        cnt = args[0]
        self.time_slider.shift_to_time(cnt)
 

    def save_image(self):
        idx = len(os.listdir('./debug'))
        cv2.imwrite('./debug/{}.png'.format(idx+1), self.pygo.img_cropped)


    def onDetectHandicap(self):
        Signals.emit(DetectHandicap)

    def videoAddNewMove(self, args):
        colour = args[0]
        ts = args[1]
        if ts is not None:
            self.time_slider.draw_stone(ts, colour)



    def __eventCoordsToGameCoords(self, event):
        x, y = event.x, event.y
        return self.__coordsToGameCoords(x,y)
    
    def __coordsToGameCoords(self, x, y):
        if self.viewVar.get() in [0,1]:
            grid = self.grid.reshape(19*19,2)
        elif self.viewVar.get() == 2:
            grid = self.grd_virtual.reshape(19*19,2)
        else:
            raise RuntimeError("Unkown View Layer")

        x_grid = np.repeat(x, 19*19)
        y_grid = np.repeat(y, 19*19)
        ref = np.stack((x_grid, y_grid)).T
        dist = np.mean((ref - grid)**2, axis=1)
        coord = np.argmin(dist)
        x_board, y_board = np.unravel_index(coord, (19,19))
        return (x_board, y_board)

    def hide_video_ui(self):
        self.time_slider_box.grid_forget()

    def show_video_ui(self):
        self.time_slider_box.grid(column=0, row=2)

 
    def leftMouseOnGOBoard(self, event):
        if self.pygo.Game.GS != GameState.NOT_STARTED:
            x_board, y_board = self.__eventCoordsToGameCoords(event)
            self.pygo.Game.setManual(x_board, y_board)

    def rightMouseOnGOBoard(self, event):
        if self.pygo.Game.GS != GameState.NOT_STARTED:
            #x_board, y_board = self.__eventCoordsToGameCoords(event)
            if self.pygo.Game.nextMove() == C2N('W'):
                self.contextMenu.entryconfig("White", state="normal")
                self.contextMenu.entryconfig("Black", state="disabled")
            elif self.pygo.Game.nextMove() == C2N('B'):
                self.contextMenu.entryconfig("White", state="disabled")
                self.contextMenu.entryconfig("Black", state="normal")
            else:
                self.contextMenu.entryconfig("White", state="normal")
                self.contextMenu.entryconfig("Black", state="normal")
            self.contextMenu.tk_popup(event.x_root, event.y_root)


    def addManualWhite(self) -> None:
        c_x = self.contextMenu.winfo_x() - self.go_board_display.winfo_rootx()
        c_y = self.contextMenu.winfo_y() - self.go_board_display.winfo_rooty()
        x,y = self.__coordsToGameCoords(c_x, c_y)
        self.pygo.Game.setManual(x,y,C2N('W'))
        ts = self.pygo.input_stream.get_time()
        self.time_slider.draw_stone(ts, 'W')


    def addManualBlack(self) -> None:
        c_x = self.contextMenu.winfo_x() - self.go_board_display.winfo_rootx()
        c_y = self.contextMenu.winfo_y() - self.go_board_display.winfo_rooty()
        x,y = self.__coordsToGameCoords(c_x, c_y)
        self.pygo.Game.setManual(x,y,C2N('B'))
        ts = self.pygo.input_stream.get_time()
        self.time_slider.draw_stone(ts, 'W')


    def addManualNone(self) -> None:
        c_x = self.contextMenu.winfo_x() - self.go_board_display.winfo_rootx()
        c_y = self.contextMenu.winfo_y() - self.go_board_display.winfo_rooty()
        x,y = self.__coordsToGameCoords(c_x, c_y)
        self.pygo.Game.setManual(x,y,C2N('E'))


    def removeManual(self) -> None:
        c_x = self.contextMenu.winfo_x() - self.go_board_display.winfo_rootx()
        c_y = self.contextMenu.winfo_y() - self.go_board_display.winfo_rooty()
        x,y = self.__coordsToGameCoords(c_x, c_y)
        self.pygo.Game.clearManual(x,y)


    def freeze(self, event=None) -> None:
        self.pygo.freeze()


    def GameTogglePauseResume(self, event=None) -> None:
        if self.pygo.Game.GS == GameState.RUNNING:
            self.GamePause()
        elif self.pygo.Game.GS == GameState.PAUSED:
            self.GameRun()
        else:
            # game is currently not started
            logging.warning('Detect the board before starting the game!')

    def onClearManualAll(self) -> None:
        self.pygo.Game.clearManualAll()

    def game_is_active(self):
        return self.pygo.Game.GS != GameState.NOT_STARTED

    def game_is_video(self):
        return self.pygo.input_stream.is_video

    def GamePause(self) -> None:
        Signals.emit(GamePause)
    
    def GameRun(self) -> None:
        Signals.emit(GameRun)
    
    def GameTreeBack(self) -> None:
        if self.game_is_active:
            Signals.emit(GameTreeBack)
        if self.game_is_video:
            Signals.emit(GamePause)

    def GameTreeForward(self) -> None:
        if self.game_is_active:
            Signals.emit(GameTreeForward)
        if self.game_is_video:
            Signals.emit(GamePause)
    
    def switchState(self, fn, name, state):
        if state.get():
            fn.enable(name)
        else:
            fn.disable(name)

    def setLogLevelInfo(self) -> None:
        logging.getLogger().setLevel(logging.INFO)

    def setLogLevelDebug(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)

    def setLogLevelDebug2(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG2)

    def setLogLevelWarn(self) -> None:
        logging.getLogger().setLevel(logging.WARN)


    def onInputDeviceChanged(self, *args):
        
        dev = args[0]
        if '/dev/video' in dev:
            #opencv only uses the number
            dev_id = int(dev[-1])
            self.pygo.input_stream.set_input_file_stream(dev_id)
            Signals.emit(GameReset, 19)
            self.hide_video_ui()
            self.go_tree_pause["state"] = "normal"
        else:
            self.video_str = fd.askopenfilename(filetypes=[('mp4', '*.mp4'),
            ])
            if self.video_str:
                self.time_slider.reset()
                self.show_video_ui()
                self.pygo.input_stream.set_input_file_stream(self.video_str)
                self.time_slider.on_update_time(self.pygo.input_stream.get_length())
                self.onGameNew()
                self.go_tree_pause["state"] = "disabled"

    def load_video(self, name):
        self.video_str = name
        self.show_video_ui()
        self.pygo.input_stream.set_input_file_stream(self.video_str)
        self.time_slider.on_update_time(self.pygo.input_stream.get_length())
        self.onGameNew()
        self.go_tree_pause["state"] = "disabled"



    def onSettings(self):
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title('Settings')
        self.settings_window.grid()
        btn1 = tk.Checkbutton(self.settings_window, 
                                text='Allow undoing moves during recording',
                                variable=self.settings['AllowUndo'],
                                onvalue=True, 
                                offvalue=False)
        btn1.grid(column=0, row=0)

        sep = ttk.Separator(self.settings_window, orient='horizontal')
        sep.grid(column=0, row=1, sticky='ew')

        lbl1 = tk.Label(self.settings_window, text="Motion Detection Agressiveness")
        lbl1.grid(column=0, row=2)

        switch_frame = tk.Frame(self.settings_window)
        switch_frame.grid(column=0, row=3)

        low_button = tk.Radiobutton(switch_frame, 
                                    text="Low", 
                                    variable=self.settings['MotionDetectionBoard'],
                                    indicatoron=False, 
                                    value=0.2, 
                                    width=8)
        med_button = tk.Radiobutton(switch_frame, 
                                    text="Medium", 
                                    variable=self.settings['MotionDetectionBoard'],
                                    indicatoron=False, 
                                    value=0.4, 
                                    width=8)
        high_button = tk.Radiobutton(switch_frame, 
                                    text="High", 
                                    variable=self.settings['MotionDetectionBoard'],
                                    indicatoron=False, 
                                    value=0.6, 
                                    width=8)
        low_button.pack(side="left")
        med_button.pack(side="left")
        high_button.pack(side="left")


        sep = ttk.Separator(self.settings_window, orient='horizontal')
        sep.grid(column=0, row=4, sticky='ew')

        lbl2 = tk.Label(self.settings_window, text="Video Input")
        lbl2.grid(column=0, row=5)


        self.v = tk.StringVar()
        self.input_devices = []
        video_ports = self.pygo.input_stream.getWorkingPorts()
        for port in video_ports:
            if port != "Select Video": 
                self.input_devices.append('/dev/video{}'.format(port))
        self.input_devices.append('Select Video')

        self.v.set(self.pygo.input_stream.current_port)

        dropdown = tk.OptionMenu(
            self.settings_window,
            self.v,
            *self.input_devices,
            command=self.onInputDeviceChanged
        ) 
        dropdown.grid(column=0, row=6)


        self.settings_window.protocol("WM_DELETE_WINDOW", self.on_settings_closing)
        Signals.emit(OnSettingsChanged, self.settings)

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
        self.pygo.input_stream.release()
        self.root.quit()
        self.root.destroy()


    def onBoardDetect(self) -> None:
        # ask for board detection
        Signals.emit(DetectBoard, self.pygo.img_cam)

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
    
    def __clear_log(self, *args):
        self.move_log.delete('1.0', tk.END)

    def onGameNew(self) -> None:
        self.__clear_log()
        Signals.emit(GameNew, 19)

    def updateLog(self, args):
        moves = args[0]
        self.move_log.delete('1.0', 'end')
        for move in moves:
            self.move_log.insert('end', move)
        self.move_log.see('end')


    def run(self) -> None:
        self.root.after(1, self.update)
        self.root.mainloop()

    def logMove(self, msg: NetMove) -> None:
        if msg is not None:
            if msg[0] == 'e':
                self.moveHistory.pop()
                self.move_log.delete('end-2l', 'end-1l')
            elif msg[0] == '':
                pass
            else:
                move = '{}: {}-{}\n'.format(msg[0], msg[1]+1, msg[2]+1)
                logging.debug('TK: new move ' + move)
                self.moveHistory.append(move)
                self.move_log.insert('end', move)
            self.move_log.see('end')  # move to the end after adding new text



    def __np2tk(self, img : Image) -> ImageTk.PhotoImage: 
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(PIL.Image.fromarray(rgb))

    def updateGrid(self, *args) -> None:
        with self.lock_grid:
            self.grid = self.pygo.Board.go_board_shifted
            self.grd_virtual = self.pygo.Board.grd_overlay


    def update(self) -> None:
        if self.weOwnControllLooop:
            self.pygo.run_once()
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

