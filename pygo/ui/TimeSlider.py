from cmath import exp
from ipaddress import collapse_addresses
import tkinter as tk
from tkinter import YView, ttk
import pdb
from turtle import back
from PIL import Image,ImageTk

class TimeSlider(tk.Frame):
    def __init__(self, master: tk.BaseWidget, *args, **kwargs):
        super().__init__(master, **kwargs)
        self.t = 0
        self.to = 2500
        self.time_ticks = 60            # place a tick every n seconds
        self.time_tick_width = 50       # space in px between px
        self.F = self.time_ticks / self.time_tick_width
        self.screen_width = master.winfo_screenwidth() / 2
        self.pad = self.screen_width //2
        self.scroll_start = self.pad
        self.scroll_end = self.pad + (self.to/self.F)
        self.lines_tick = []
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
                                        scrollregion=(self.scroll_start, 0, self.scroll_end, 0))
        self.inner_canvas.configure(xscrollcommand=self.scrollbar.set)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self._draw()
        self.scrollbar.grid(column=0, row=0, sticky=tk.W+tk.E)
        self.inner_canvas.grid(column=0, row=1, sticky=tk.W+tk.E)
        self.ref.grid(column=0, row=2, sticky=tk.W+tk.E)
        

        w2 = int(self.winfo_width()/2)
        self.middle_line = self.ref.create_line(w2, 0, w2, 20, fill='#FF0000')
        self.label = self.ref.create_text(w2, 25, text=self.t)


        self.inner_canvas.bind("<ButtonPress-1>", self.scroll_begin)
        self.inner_canvas.bind("<B1-Motion>", self.scroll_move)
        self.inner_canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.inner_canvas.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.inner_canvas.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up
        self.bind("<Configure>", self.on_resize)

    def update_scrollregion(self, win_width=None):    
        if win_width is not None:
            w2 = win_width / 2
        else:
            w2 = int(self.winfo_width() / 2)

        self.scroll_end = self.pad + (self.to/self.F)

        (ss, se) = self.scrollbar.get()
        sw = (se-ss) 
        max_ = 1.0 - sw
        fraction = (self.t / (self.to/max_))  #* self.F
        
        self.inner_canvas.config(scrollregion=(self.scroll_start-w2, 0, self.scroll_end+w2, 0)) 
        #self.inner_canvas.xview_moveto(fraction)
        self.set_time_seconds(self.t)
 

    def on_resize(self, event=None):
        if event is not None:
            w = event.width
        else:
            w = self.winfo_width()
        w2 = int(w/2)
        self.ref.coords(self.middle_line, w2, 0, w2, 20)
        self.ref.coords(self.label, w2, 25)
        #self.update_scrollregion(w)
        self.scroll_end = self.pad + (self.to/self.F)
        self.inner_canvas.config(scrollregion=(self.scroll_start-w2, 0, self.scroll_end+w2, 0)) 
        self.set_time_seconds(self.t)

 
        #(l,r) = self.scrollbar.get()

    def set_time_fraction(self, *args):
        (ss, se) = self.scrollbar.get()
        sw = (se-ss) 
        fraction = float(args[1])
        fraction = min(max(0,fraction),1.0-sw)
        self.inner_canvas.xview(args[0], args[1])
        max_ = 1.0 - sw
        t_rescaled = self.to / max_
        time = int(fraction * t_rescaled)
        #pdb.set_trace()
        #fraction  = float(fraction) + sw
        self.update_time(time)

    def _get_time_from_scrollbar(self) -> int:
        (ss, se) = self.scrollbar.get()
        sw = (se-ss) 
        fraction = ss
        fraction = min(max(0,fraction),1.0-sw)
        max_ = 1.0 - sw
        t_rescaled = self.to / max_
        time = int(fraction * t_rescaled)
        return time
 

    def set_time_seconds(self, time):
        fraction = time / self.to
        self.inner_canvas.xview_moveto(fraction)
        time = min(max(time, 0), self.to)
        self.update_time(time)


    def update_time(self, time=None):
        if time is None:
            time = self._get_time_from_scrollbar()
            #time = int(float(self.scrollbar.get()[0]) * self.to)
            #print(time)

        self.t = time
        self.ref.itemconfigure(self.label, text=str(time))

    def scroll_begin(self, event):
        self.inner_canvas.scan_mark(event.x, 0)
        self.update_time()

    def scroll_move(self, event):
        self.inner_canvas.scan_dragto(event.x, 0, gain=1)
        self.update_time()

    def wheel(self, event):
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            self.F *= 1.25
            scale *= 1.25
        if event.num == 4 or event.delta == 120:  # scroll up
            self.F /= 1.25
            scale /= 1.25
        #x = self.inner_canvas.canvasx(event.x)
        #y = self.inner_canvas.canvasy(event.y)
        #self.inner_canvas.scale('all', x, y, scale, 1.0)
        self.scale_timeline()
        self._scale_stones()
        self.update_scrollregion()

    def scale_timeline(self):
        h = 150
        h2 = int(h/2)
        bot = 15
        self.inner_canvas.coords('timeline', self.pad, h2, self.pad+(self.to/self.F), h2) 
        self.inner_canvas.coords("line_start", self.pad-2, 3*bot, self.pad-2, h-(3*bot))
        self.inner_canvas.coords("line_end0", self.pad+(self.to/self.F), 3*bot, self.pad+(self.to/self.F), h-(3*bot))
        self.inner_canvas.coords("line_end1", self.pad+(self.to/self.F)+2, 3*bot, self.pad+(self.to/self.F)+2, h-(3*bot))

        for sec, line_id in self.lines_tick:
            self.inner_canvas.coords(line_id, 
                self.pad+sec/self.F, 
                3*bot, 
                self.pad+sec/self.F, 
                h-(3*bot))

        for sec, text_id in self.text_tick:
            self.inner_canvas.coords(text_id, 
                self.pad+sec/self.F, 
                h-(3*bot)+5)




    def _draw(self):
        h = 150
        h2 = int(h/2)
        bot = 15
        self.inner_canvas.create_line(self.pad, h2, self.pad+(self.to/self.F), h2, tags="timeline")
        # start/end lines
        self.inner_canvas.create_line(self.pad-2, 3*bot, self.pad-2, h-(3*bot), tags='line_start')
        self.inner_canvas.create_line(self.pad+(self.to/self.F), 3*bot, self.pad+(self.to/self.F), h-(3*bot), tags="line_end0")
        self.inner_canvas.create_line(self.pad+(self.to/self.F)+2, 3*bot, self.pad+(self.to/self.F)+2, h-(3*bot), tags="line_end1")

        for sec in range(0, self.to, self.time_ticks):
            id = self.inner_canvas.create_line(self.pad+sec/self.F, 3*bot, self.pad+sec/self.F, h-(3*bot))
            self.lines_tick.append((sec,id))
            if sec % 300 == 0:
                id = self.inner_canvas.create_text(self.pad+sec/self.F, h-(3*bot)+5, text=sec)
                self.text_tick.append((sec, id))

    def _scale_stones(self):
        for (id, time) in self.stones:
            x0, y0, x1, y1 = self._get_stone_coordinates(time)
            self.inner_canvas.coords(id, x0, y0, x1, y1)
            
    def _get_stone_coordinates(self, time):
        width = 15
        h = 150
        h2 = int(h/2)
 
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
