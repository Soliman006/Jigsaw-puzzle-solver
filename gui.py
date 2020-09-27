import tkinter as tk


class GUI:
    """GUI for interacting with Anaerobic Diestion model."""

    def __init__(self):
        self.data_init()
        self.main_window_init()
        self.window.mainloop()

    def data_init(self):
        self.p = 5

    def main_window_init(self):
        self.window = tk.Tk()
        self.window.title("Anaerobic Digestion Model")
        self.window.geometry("900x900+10+20")
        self.main_controls(10, 10)
        self.output_selection(200, 200)

    def main_controls(self, x0, y0):
        self.button_run = tk.Button(self.window, text="RUN", fg="black", command=self.run)
        self.button_run.place(x=x0, y=y0)

        self.button_load = tk.Button(self.window, text="LOAD", fg="black", command=self.load)
        self.button_load.place(x=x0+50, y=y0)

        self.button_quit = tk.Button(self.window, text="QUIT", fg="red", command=self.window.destroy)
        self.button_quit.place(x=x0+100, y=y0)

    def output_selection(self, x0, y0):
        self.v1 = tk.BooleanVar()
        self.v2 = tk.BooleanVar()
        self.checkbox_1 = tk.Checkbutton(self.window, text="Option 1", variable=self.v1)
        self.checkbox_2 = tk.Checkbutton(self.window, text="Option 2", variable=self.v2)
        self.checkbox_1.place(x=x0+10, y=y0+10)
        self.checkbox_2.place(x=x0+10, y=y0+50)

    def run(self):
        print("Running Simulation")
        print("v1:", self.v1.get(), "v2:", self.v2.get())

    def load(self):
        print("Loading")
