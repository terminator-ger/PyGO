from cmath import exp
from ipaddress import collapse_addresses
import tkinter as tk
from tkinter import Scale, YView, ttk
import pdb
from turtle import back
from PIL import Image,ImageTk
import time

class TimeSlider(tk.Frame):
    def __init__(self, master: tk.BaseWidget, *args, **kwargs):
        super().__init__(master, **kwargs)
        self.scale = 1.0
        self.t = 0
        self.to = 2500
        self.time_ticks = 60            # place a tick every n seconds
        self.time_tick_width = 50       # space in px between px
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

        self.scrollbar = tk.Scrollbar(
            self, 
            orient="horizontal",
            command=self.set_time_fraction
        )
        self.ref = tk.Canvas(self,
                            height=35)
        self.inner_canvas = tk.Canvas(self, 
                                        height=135,
                                        scrollregion=(self.scroll_start, 0, 
                                                      self.scroll_end, 0))
        self.inner_canvas.configure(xscrollcommand=self.scrollbar.set)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.scrollbar.grid(column=0, row=0, sticky=tk.W+tk.E)
        self.inner_canvas.grid(column=0, row=1, sticky=tk.W+tk.E)
        self.ref.grid(column=0, row=2, sticky=tk.W+tk.E)
        

        self.middle_line = self.ref.create_line(self.w2, 0, self.w2, 20, fill='#FF0000')
        
        self.label = self.ref.create_text(self.w2, 25, text=time.strftime('%H:%M:%S', time.gmtime(self.t)))


        self.inner_canvas.bind("<ButtonPress-1>", self.scroll_begin)
        self.inner_canvas.bind("<B1-Motion>", self.scroll_move)
        self.inner_canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.inner_canvas.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.inner_canvas.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up
        self.bind("<Configure>", self.on_resize)
        self._draw()
 

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

        self.shift_to_time(self.t)

 

    def set_time_fraction(self, *args):
        (ss, se) = self.scrollbar.get()
        sw = (se-ss) 
        fraction = float(args[1])
        #fraction = min(max(0,fraction),1.0-sw)
        self.inner_canvas.xview(args[0], args[1])
        max_ = 1.0 - sw
        t_rescaled = self.to / max_
        time = fraction * t_rescaled
        time = max(min(time, self.to),0)
        self.shift_to_time(time)

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
    
    def shift_to_time(self, t: float) -> None:
        '''
        receives a timestamp in seconds, adjusts the timeline to match the new position
        '''
        # update label
        if self.t != t:
            self.t = t
            t_str = time.strftime('%H:%M:%S', time.gmtime(int(self.t)))
            self.ref.itemconfigure(self.label, text=t_str)

        sw = (2*self.w2) / (self.scroll_end - self.scroll_start)
        max_ = 1.0 - sw
        t_end_rescaled = self.to / max_
        rel_pos = self.t 
        if rel_pos != 0:
            rel_pos = rel_pos / t_end_rescaled
        self.inner_canvas.xview_moveto(rel_pos)


    def update_time_from_scrollbar(self):
        ftime = self._get_time_from_scrollbar()
        self.t = ftime
        t_str = time.strftime('%H:%M:%S', time.gmtime(int(self.t)))
        self.ref.itemconfigure(self.label, text=t_str)

    def scroll_begin(self, event):
        self.inner_canvas.scan_mark(event.x, 0)
        #self.update_time_from_scrollbar()

    def scroll_move(self, event):
        self.inner_canvas.scan_dragto(event.x, 0, gain=1)
        self.update_time_from_scrollbar()

    def wheel(self, event):
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            self.scale += 0.05
        if event.num == 4 or event.delta == 120:  # scroll up
            self.scale -= 0.05

        self.scale = max(min(self.scale, 1.75), 0.25)

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
 
        self.inner_canvas.config(scrollregion=(self.scroll_start, 0, 
                                               self.scroll_end, 0)) 
 
        self.shift_to_time(self.t)




    def _draw(self):
        h = 150
        h2 = h/2
        bot = 15

        self.inner_canvas.create_line(self.data_start, h2, 
                                      self.data_end, h2, tags="timeline")
        # start/end lines
        self.inner_canvas.create_line(self.data_start-2, 3*bot, 
                                        self.data_start-2, h-(3*bot), tags='line_start')
        self.inner_canvas.create_line(self.data_end, 3*bot, 
                                       self.data_end, h-(3*bot), tags="line_end0")
        self.inner_canvas.create_line(self.data_end+2, 3*bot, 
                                        self.data_end+2, h-(3*bot), tags="line_end1")

        for sec in range(0, self.to, self.time_ticks):
            # 5 min ticks
            if sec % 300 == 0: 
                time_str = time.strftime('%H:%M:%S', time.gmtime(sec))
                id = self.inner_canvas.create_text(self.data_start+(sec/self.F), h-(3*bot)+5, text=time_str)
                self.text_tick.append((sec, id))
                id = self.inner_canvas.create_line(self.data_start + (sec/self.F), 3*bot, 
                                                self.data_start + (sec/self.F), h-(3*bot))
                self.lines_tick_major.append((sec,id))
 
            # 1 min ticks
            elif sec % 60 == 0: # 1 min ticks
                id = self.inner_canvas.create_line(self.data_start + (sec/self.F), 4*bot, 
                                                self.data_start + (sec/self.F), h-(4*bot))
                self.lines_tick_minor.append((sec,id))
 

    def _scale_stones(self):
        for (id, time) in self.stones:
            x0, y0, x1, y1 = self._get_stone_coordinates(time)
            self.inner_canvas.coords(id, x0, y0, x1, y1)
            
    def _get_stone_coordinates(self, time):
        width = 15
        h = 150
        h2 = h/2
 
        x0 = self.pad + (time/self.F) - (width)
        x1 = self.pad + (time/self.F) + (width)
        y0 = h2 - (width)
        y1 = h2 + (width)
        return (x0, y0, x1, y1)


    def draw_stone(self, time: int, color: str =None):
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
