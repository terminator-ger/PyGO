import time
import tkinter as tk
from PIL import Image, ImageTk

from pygo.core import PyGO
from pygo.Signals import *
from pygo.Game import GameState

class TimeSlider(tk.Frame):
    def __init__(self, master: tk.BaseWidget, pygo: PyGO, *args, **kwargs):
        super().__init__(master, **kwargs)
        self.scale = 1.0
        self.pygo = pygo
        # we have two times one for where the current playback is and 
        # one for the cursors position
        self.t = 0          # position of black cursor
        self.t_cursor = 0   # position of red cursor
        self.to = 240
        self.time_ticks = 60            # place a tick every n seconds
        self.time_tick_width = 80       # space in px between ticks
        self.ttws = self.time_tick_width * self.scale
        self.F = self.time_ticks / self.ttws
        self.screen_width = master.winfo_screenwidth() / 2
        self.w2 = self.winfo_width()/2
        # we pad the dataline with a sourrinding space of screenwidth /2 to ensure enough space on all resizes
        self.pad = self.screen_width

        self.data_start = self.pad
        self.data_end = self.pad + (self.to / self.F)

        self.scroll_start = self.pad - self.w2
        self.scroll_end = self.pad + (self.to/self.F) + self.w2

        self.lines_tick_major = []
        self.lines_tick_minor = []
        self.text_tick = []
        self.stones = []

        self.ref = tk.Canvas(self,
                            height=35)
        self.inner_canvas = tk.Canvas(self, 
                                        height=135,
                                        scrollregion=(self.scroll_start, 0, 
                                                      self.scroll_end, 0))
        self.scrollbar = tk.Scrollbar(
            self, 
            orient="horizontal",
            command=self.set_time_fraction
            #command=self.inner_canvas.xview
        )
 
        self.inner_canvas.configure(xscrollcommand=self.scrollbar.set)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.scrollbar.grid(column=0, row=0, sticky=tk.W+tk.E)
        self.inner_canvas.grid(column=0, row=1, sticky=tk.W+tk.E)
        self.ref.grid(column=0, row=2, sticky=tk.W+tk.E)
        self.navigation_buttons = tk.LabelFrame(self)
        self.navigation_buttons.grid(column=0, row=3)


        self.img_fb     = Image.open("img/rewind2.png")
        self.img_fb10   = Image.open('img/rewind.png')
        self.img_pr     = Image.open('img/play-red.png')
        self.img_p      = Image.open('img/play-button.png')
        self.img_fw     = Image.open('img/fast-forward2.png')
        self.img_fw10   = Image.open('img/fast-forward.png')
        self.img_pause  = Image.open('img/pause.png')

        self.img_fb      = self.img_fb.resize((32,32), Image.ANTIALIAS) 
        self.img_fb10    = self.img_fb10.resize((32,32), Image.ANTIALIAS) 
        self.img_pr      = self.img_pr.resize((32,32), Image.ANTIALIAS) 
        self.img_p       = self.img_p.resize((32,32), Image.ANTIALIAS) 
        self.img_fw      = self.img_fw.resize((32,32), Image.ANTIALIAS) 
        self.img_fw10    = self.img_fw10.resize((32,32), Image.ANTIALIAS) 
        self.img_pause   = self.img_pause.resize((32,32), Image.ANTIALIAS)

        self.img_btn_fb =    ImageTk.PhotoImage(self.img_fb)
        self.img_btn_fb10 =  ImageTk.PhotoImage(self.img_fb10)
        self.img_btn_pr =    ImageTk.PhotoImage(self.img_pr)
        self.img_btn_p =     ImageTk.PhotoImage(self.img_p)
        self.img_btn_fw =    ImageTk.PhotoImage(self.img_fw)
        self.img_btn_fw10 =  ImageTk.PhotoImage(self.img_fw10)
        self.img_btn_pause = ImageTk.PhotoImage(self.img_pause)

        self.nav_btn_fb         = tk.Button(self.navigation_buttons, image=self.img_btn_fb,   width=35, command=self.seek_back)
        self.nav_btn_10fb       = tk.Button(self.navigation_buttons, image=self.img_btn_fb10, width=35, command=self.seek_back10)
        self.nav_btn_play_red   = tk.Button(self.navigation_buttons, image=self.img_btn_pr,   width=35, command=self.play_pause_red)
        self.nav_btn_play_black = tk.Button(self.navigation_buttons, image=self.img_btn_p,    width=35, command=self.play_pause)
        self.nav_btn_fw         = tk.Button(self.navigation_buttons, image=self.img_btn_fw,   width=35, command=self.seek_forward)
        self.nav_btn_10fw       = tk.Button(self.navigation_buttons, image=self.img_btn_fw10, width=35, command=self.seek_forward10)


        self.nav_btn_fb.grid(column=0, row=0) 
        self.nav_btn_10fb.grid(column=1, row=0)
        self.nav_btn_play_red.grid(column=2, row=0)
        self.nav_btn_play_black.grid(column=3, row=0) 
        self.nav_btn_10fw.grid(column=4, row=0) 
        self.nav_btn_fw.grid(column=5, row=0) 

        self.middle_line = self.ref.create_line(self.w2, 0, self.w2, 20, fill='#FF0000', width=2)
        
        self.label = self.ref.create_text(self.w2, 25, text=time.strftime('%H:%M:%S', time.gmtime(0)))


        self.seek_job = None
        self.inner_canvas.bind("<ButtonPress-1>", self.scroll_begin)
        self.inner_canvas.bind("<B1-Motion>", self.scroll_move)
        self.inner_canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.inner_canvas.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.inner_canvas.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up
        self.bind("<Configure>", self.on_resize)
        self._draw()
        self._update_buttons(0)
        self.shift_to_time(0)


    def seek(self, event):
        if self.seek_job:
            self.after_cancel(self.seek_job)
        self.seek_job = self.after(500, self._update_buttons, self.t_cursor, True)


    def seek_back(self):
        Signals.emit(InputBackward)
        val = self.pygo.input_stream.get_time()
        self.shift_to_time(val, cursor_only=True)
 
    def seek_back10(self):
        Signals.emit(InputBackward10)
        val = self.pygo.input_stream.get_time()
        self.shift_to_time(val, cursor_only=True)
 
    def seek_forward(self):
        Signals.emit(InputForward)
        val = self.pygo.input_stream.get_time()
        self.shift_to_time(val, cursor_only=True)
    
    def seek_forward10(self):
        Signals.emit(InputForward10)
        val = self.pygo.input_stream.get_time()
        self.shift_to_time(val, cursor_only=True)
 

    def play_pause_red(self):
        val = self.t_cursor
        self.shift_to_time(val)
        self._update_buttons(val)

    def play_pause(self):
        val = self.t
        self.shift_to_time(val)
        self._update_buttons(val)


    def _update_buttons(self, val=None, keep_paused=False) -> None:
        if not keep_paused:
            if self.pygo.Game.GS == GameState.RUNNING:
                self.nav_btn_play_red["state"] = "normal" 
                self.nav_btn_play_black.configure(image=self.img_btn_p)
                Signals.emit(GamePause)

            elif self.pygo.Game.GS == GameState.PAUSED:
                self.nav_btn_play_red["state"] = "disabled" 
                self.nav_btn_play_black.configure(image=self.img_btn_pause)
                if val is not None:
                    Signals.emit(InputStreamSeek, val)
                    Signals.emit(PreviewNextFrame)
                Signals.emit(GameRun)

            elif self.pygo.Game.GS == GameState.NOT_STARTED:
                self.nav_btn_play_red["state"] = "disabled"
                self.nav_btn_play_black.configure(image=self.img_btn_p)

        else:
            # for usage with slider/scrollbar, keep the game in paused state and preview
            # the frame under the cursor in paused state
            self.nav_btn_play_red["state"] = "normal" 
            self.nav_btn_play_black.configure(image=self.img_btn_p)
            Signals.emit(GamePause)
            if val is not None:
                Signals.emit(InputStreamSeek, val)
                Signals.emit(PreviewNextFrame)



    def _set_frame_pos(self) -> None:
        val = self.t_cursor
        Signals.emit(InputStreamSeek, val)


    def on_update_time(self, total_seconds: float) -> None:
        self.to = total_seconds
        self.t_cursor = 0
        self._scale_ticks()
        self._draw()
        self.shift_to_time(0)

    def get_cursor_time(self) -> float:
        return self.t_cursor
 

    def on_resize(self, event=None):
        '''
            resizes the window
        '''

        if event is not None:
            w = event.width
        else:
            w = self.winfo_width()
        self.w2 = w/2      # update new window half width

        self._scale_ticks()
        #update pointer to current timepoint
        self.ref.coords(self.middle_line, self.w2, 0, self.w2, 20)
        self.ref.coords(self.label, self.w2, 25)

        # wider window -> adjust scrollbar
        self.inner_canvas.config(scrollregion=(self.scroll_start, 0,
                                               self.scroll_end, 0))


    def set_time_fraction(self, *args):
        self.inner_canvas.xview(*args)
        (ss, se) = self.scrollbar.get()
        sw = (se-ss) 
        fraction = float(args[1])
        max_ = 1.0 - sw
        t_rescaled = self.to / max_
        time = fraction * t_rescaled
        time = max(min(time, self.to),0)
        self.shift_to_time(time, cursor_only=True)
        self.seek()

    def _get_time_from_scrollbar(self) -> int:
        (ss, se) = self.scrollbar.get()
        sw = (se-ss) 
        fraction = ss
        #fraction = min(max(0,fraction),1.0-sw)
        max_ = 1.0 - sw
        t_rescaled = self.to / max_
        time = fraction * t_rescaled
        time = max(min(time, self.to),0)
        return time
    
    def shift_to_time(self, new_time: float, cursor_only: bool =False) -> None:
        '''
        receives a timestamp in seconds, adjusts the timeline to match the new position
        '''
        # update label
        if cursor_only:
            self.t_cursor = new_time
        else:
            self.t = new_time
            self.t_cursor = new_time
        
        t_str = time.strftime('%H:%M:%S', time.gmtime(int(new_time)))
        self.ref.itemconfigure(self.label, text=t_str)

        sw = (2*self.w2) / (self.scroll_end - self.scroll_start)
        max_ = 1.0 - sw
        t_end_rescaled = self.to / max_
        h = 150
        bot = 15
        t = 3*bot
        b = h-(3*bot)

        if not cursor_only:
            self.inner_canvas.coords("cur_time", self.data_start+(new_time/self.F), t, 
                                            self.data_start+(new_time/self.F), b)
        
        if new_time != 0:
            new_time /= t_end_rescaled
        
        logging.debug2("Time update, moving to {}".format(new_time))
        self.inner_canvas.xview_moveto(new_time)


    def update_time_from_scrollbar(self):
        ftime = self._get_time_from_scrollbar()
        self.t_cursor = ftime
        t_str = time.strftime('%H:%M:%S', time.gmtime(int(self.t_cursor)))
        self.ref.itemconfigure(self.label, text=t_str)

    def scroll_begin(self, event):
        self._update_buttons(keep_paused=True)
        self.inner_canvas.scan_mark(event.x, 0)
        # pause game


    def scroll_move(self, event):
        self.inner_canvas.scan_dragto(event.x, 0, gain=1)
        self.update_time_from_scrollbar()
        self.seek(event)


    def wheel(self, event):
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            self.scale += 0.05
        if event.num == 4 or event.delta == 120:  # scroll up
            self.scale -= 0.05

        self.scale = max(min(self.scale, 1.9), 0.1)

        self._scale_ticks()
        self.scale_timeline()
        self._scale_stones()
        self.update_scrollregion()


    def scale_timeline(self):
        '''
            Redraws the timeline with the updated tick width
        '''
        h = 150
        h2 = h/2
        bot = 15
        t = 3*bot
        b = h-(3*bot)
        t_minor = 4*bot
        b_minor = h-(4*bot)
        start = self.data_start
        end = self.data_end
        ipt_time = self.pygo.input_stream.get_time()
        T = ipt_time / self.F

        self.inner_canvas.coords("cur_time", start+T, t, start+T, b)
        self.inner_canvas.coords('timeline',    start,   h2, end,     h2)
        self.inner_canvas.coords("line_start",  start-2, t,  start-2, b)
        self.inner_canvas.coords("line_end0",   end,     t,  end,     b)
        self.inner_canvas.coords("line_end1",   end+2,   t,  end+2,   b)

        for sec, line_id in self.lines_tick_major:
            self.inner_canvas.coords(line_id,  
                self.pad+sec/self.F,  t,
                self.pad+sec/self.F,  b)

        for sec, line_id in self.lines_tick_minor:
            self.inner_canvas.coords(line_id,  
                self.pad+sec/self.F,  t_minor,
                self.pad+sec/self.F,  b_minor)


        for sec, text_id in self.text_tick:
            self.inner_canvas.coords(text_id, 
                self.pad+sec/self.F,  b+5)


    def _scale_ticks(self):
        self.ttws = self.time_tick_width * self.scale
        self.F = self.time_ticks / self.ttws

        self.data_start = self.pad
        self.data_end = self.pad + (self.to / self.F)

        self.scroll_start = self.pad - self.w2
        self.scroll_end = self.pad + (self.to/self.F) + self.w2


    def update_scrollregion(self, win_width=None):    
        if win_width is not None:
            self.w2 = win_width / 2
        else:
            self.w2 = self.winfo_width() / 2

        logging.debug2("Scrolling to {}".format(self.scroll_start))

        self.inner_canvas.config(scrollregion=(self.scroll_start, 0, 
                                               self.scroll_end, 0)) 
 

    def _draw(self):
        h = 150
        h2 = h/2
        bot = 15

        t_major = 3*bot
        b_major = h-(t_major)

        t_minor = 4*bot
        b_minor = h-(t_minor)
        T = self.pygo.input_stream.get_time() / self.F
        self.inner_canvas.create_line(self.data_start+T, t_major, 
                                      self.data_start+T, b_major, tags="cur_time",
                                      width=2)

        self.inner_canvas.create_line(self.data_start, h2, 
                                      self.data_end, h2, tags="timeline")
        # start/end lines
        self.inner_canvas.create_line(self.data_start-2, t_major, 
                                        self.data_start-2, b_major, tags='line_start')
        self.inner_canvas.create_line(self.data_end, t_major, 
                                       self.data_end, b_major, tags="line_end0")
        self.inner_canvas.create_line(self.data_end+2, t_major, 
                                        self.data_end+2, b_major, tags="line_end1")

        for sec in range(0, int(self.to), self.time_ticks):
            # 5 min ticks
            if sec % 300 == 0: 
                time_str = time.strftime('%H:%M:%S', time.gmtime(sec))
                id = self.inner_canvas.create_text(self.data_start+(sec/self.F), b_major+5, text=time_str)
                self.text_tick.append((sec, id))
                id = self.inner_canvas.create_line(self.data_start + (sec/self.F), t_major, 
                                                self.data_start + (sec/self.F), b_major)
                self.lines_tick_major.append((sec,id))
 
            # 1 min ticks
            elif sec % 60 == 0: # 1 min ticks
                id = self.inner_canvas.create_line(self.data_start + (sec/self.F), t_minor, 
                                                self.data_start + (sec/self.F), b_minor)
                self.lines_tick_minor.append((sec,id))
        self.inner_canvas.xview_moveto(0)

    def _scale_stones(self):
        for (id, time) in self.stones:
            x0, y0, x1, y1 = self._get_stone_coordinates(time)
            self.inner_canvas.coords(id, x0, y0, x1, y1)
            
    def _get_stone_coordinates(self, time):
        width = int(10 * self.scale)
        h = 150
        h2 = h/2
 
        x0 = self.pad + (time/self.F) - (width)
        x1 = self.pad + (time/self.F) + (width)
        y0 = h2 - (width)
        y1 = h2 + (width)
        return (x0, y0, x1, y1)


    def draw_stone(self, time: int, color: str =None):
        if not color.upper() in ['B', 'W']:
            return

        x0, y0, x1, y1 = self._get_stone_coordinates(time)
        if color.upper() == 'B':
            fill = '#000000'
        elif color.upper() == 'W':
            fill = '#FFFFFF'

        id = self.inner_canvas.create_oval(x0, y0, x1, y1, fill=fill)
        self.stones.append((id, time))














if __name__ == '__main__':
    root = tk.Tk()
    root.title('TimeSlider')
    tk.Grid.rowconfigure(root, 0, weight=1)
    tk.Grid.columnconfigure(root, 0, weight=1)
    ts = TimeSlider(root)
    ts.grid(column=0, row=0, sticky=tk.W+tk.E)
    ts.draw_stone(150, 'W')
    root.mainloop()
