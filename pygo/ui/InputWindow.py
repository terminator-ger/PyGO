import tkinter as tk
from pygo.core import PyGO
from pygo.ui.pygotk import PyGOTk

class InputWindow(tk.Toplevel):
    def __init__(self, root, pygo: PyGO, parent: PyGOTk) -> None:
        tk.Toplevel.__init__(self, root)

        self.pygo = pygo
        self.parent = parent
        self.title('Select input')
        self.grid()
        self.video_str = tk.StringVar()

        packlist = []
        self.v = tk.IntVar()
        self.v.set(0)  # initializing the choice, i.e. Python
        video_ports = self.pygo.input_stream.getWorkingPorts()
        video_ports.append('Select Video')

        self.input_devices = []
        for i,port in enumerate(video_ports):
            if port != "Select Video": 
                self.input_devices.append(('/dev/video{}'.format(port), i))
            else:
                self.input_devices.append((port, i))

        self.Tbox = tk.Text(self, height=1, width=30)
        self.Tbox.insert(tk.END,'Select Video')
        btn = tk.Button(self, 
                        command=self.parent.onVideoFileOpen)
        

        packlist.append(self.Tbox)
        packlist.append(btn)


        tk.Label(self, 
                text="Choose Input",
                justify = tk.LEFT,
                padx = 20).pack()

        for txt, val in self.input_devices:
            tk.Radiobutton(self, 
                        text=txt,
                        padx = 20, 
                        variable=self.v, 
                        command=self.parent.onInputDeviceChanged if txt != 'Select Video' else self.onVideoFileOpen,
                        value=val).pack(anchor=tk.W)

        [p.pack() for p in packlist]

    def onVideoFileOpen(self) -> None:
        self.video_str = fd.askopenfilename()

        if self.video_str:
            self.Tbox.delete('1.0', tk.END)
            self.Tbox.insert(tk.end, self.video_str)
            self.pygo.input_stream.set_input_file_stream(self.video_str)
            self.parent.onGameNew()

    def onInputChange(self) -> None:

        self.input_window = tk.Toplevel(self.root)
        self.input_window.title('Select input')
        self.input_window.grid()
        self.video_str = tk.StringVar()

        packlist = []
        self.v = tk.IntVar()

        self.v.set(0)  # initializing the choice, i.e. Python
        video_ports = self.pygo.input_stream.getWorkingPorts()
        video_ports.append('Select Video')

        self.input_devices = []
        for i,port in enumerate(video_ports):
            if port != "Select Video": 
                self.input_devices.append(('/dev/video{}'.format(port), i))
            else:
                self.input_devices.append((port, i))

        Tbox = tk.Text(self.input_window, height=1, width=30)
        Tbox.insert(tk.END,'Select Video')
        btn = tk.Button(self.input_window, 
                        command=self.onVideoFileOpen)
        

        packlist.append(Tbox)
        packlist.append(btn)


        tk.Label(self, 
                text="Choose Input",
                justify = tk.LEFT,
                padx = 20).pack()

        for txt, val in self.input_devices:
            tk.Radiobutton(self, 
                        text=txt,
                        padx = 20, 
                        variable=self.v, 
                        command=self.onInputDeviceChanged if txt != 'Select Video' else self.onVideoFileOpen,
                        value=val).pack(anchor=tk.W)

        [p.pack() for p in packlist]


    def onInputDeviceChanged(self):
        dev, i = self.input_devices[self.v.get()]
        self.pygo.input_stream.set_input_file_stream(dev)



