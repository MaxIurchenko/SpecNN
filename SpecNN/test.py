import tkinter as tk
from tkinter import filedialog, colorchooser, ttk
import spectral
import numpy as np
from PIL import Image, ImageTk


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Alloy Spectre Detection with Neural Networks")

        # Create the main frame
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.frame.grid_rowconfigure(0, weight=0)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # Top menu
        self.top_menu = tk.Label(self.frame)
        self.top_menu.config(justify='left', background="lightgray", height=2)
        self.top_menu.grid(row=0, column=0, columnspan=2, sticky='nwe')

        self.open_file_button = tk.Button(self.top_menu, text="Open HDR", command=self.open_file)
        self.open_file_button.pack(side="left", padx=0, pady=0)

        # Image display
        self.label_image = tk.Canvas(self.frame, bg="white")
        self.label_image.grid(row=1, column=0, sticky="news")

        # Bind the resize event and mouse events to the canvas
        self.label_image.bind("<Configure>", self.on_resize)

        # Right label with parameters
        self.right_label = tk.Label(self.frame)
        self.right_label.config(justify='left', background='white', width=40)
        self.right_label.grid(row=1, column=1, sticky='ns')

        self.rect_table = ttk.Treeview(self.right_label, columns=("x1", "y1", "x2", "y2", "color"), show="headings")
        self.rect_table.heading("x1", text="X1")
        self.rect_table.heading("y1", text="Y1")
        self.rect_table.heading("x2", text="X2")
        self.rect_table.heading("y2", text="Y2")
        self.rect_table.heading("color", text="Color")
        self.rect_table.grid(row=0, column=0, sticky='nw')

        self.delete_button = tk.Button(self.right_label, text="Delete Selected", command=self.delete_rectangle)
        self.delete_button.grid(row=1, column=0, sticky="nwe")

        # Variables
        self.spec_image = None
        self.rgb_image = None
        self.tk_image = None
        self.start_x = None
        self.start_y = None
        self.rectangles = []
        self.current_rectangle = None
        self.zoom_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Rectangle
        self.label_image.bind("<ButtonPress-1>", self.start_draw)
        self.label_image.bind("<B1-Motion>", self.draw_rectangle)
        self.label_image.bind("<ButtonRelease-1>", self.complete_rectangle)

    def open_file(self):
        file_path = filedialog.askopenfilename(title='Open File', filetypes=(("Spectra File", "*.hdr"), ("All files", "*.*")))
        if file_path.endswith('.hdr'):
            self.spec_image = spectral.open_image(file_path).load()
            metadata = self.spec_image.metadata

            default_bands = metadata.get("default bands", [])
            if default_bands:
                default_bands = [int(band) for band in default_bands]

            bands = default_bands
            rgb_image = np.zeros((int(metadata["lines"]), int(metadata["samples"]), 3), dtype=np.float32)
            for i, band_index in enumerate(bands):
                band_data = np.squeeze(self.spec_image[:, :, band_index])
                max_value = np.amax(band_data)
                if max_value > 0:
                    rgb_image[:, :, i] = band_data / max_value

            self.rgb_image = (rgb_image * 255).astype(np.uint8)
            self.zoom_factor = 1.0
            self.display_image()

    def display_image(self):
        pil_image = Image.fromarray(self.rgb_image)
        img_width, img_height = pil_image.size
        new_width = int(img_width * self.zoom_factor)
        new_height = int(img_height * self.zoom_factor)
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.label_image.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)
        self.update_canvas_size()

        for rect_data in self.rectangles:
            x1, y1, x2, y2, color = rect_data
            self.label_image.create_rectangle(x1 * self.zoom_factor + self.offset_x,
                                              y1 * self.zoom_factor + self.offset_y,
                                              x2 * self.zoom_factor + self.offset_x,
                                              y2 * self.zoom_factor + self.offset_y,
                                              outline=color)

    def update_canvas_size(self):
        if self.tk_image:
            self.label_image.config(width=self.tk_image.width(), height=self.tk_image.height())

    def start_draw(self, event):
        self.start_x = (event.x - self.offset_x) / self.zoom_factor
        self.start_y = (event.y - self.offset_y) / self.zoom_factor
        self.current_rectangle = self.label_image.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

    def draw_rectangle(self, event):
        if self.current_rectangle:
            self.label_image.coords(self.current_rectangle, self.start_x * self.zoom_factor + self.offset_x,
                                     self.start_y * self.zoom_factor + self.offset_y, event.x, event.y)

    def complete_rectangle(self, event):
        if self.current_rectangle:
            x1, y1, x2, y2 = self.label_image.coords(self.current_rectangle)
            color = self.choose_color()
            if color:
                self.label_image.itemconfig(self.current_rectangle, outline=color)
                self.rectangles.append(((x1 - self.offset_x) / self.zoom_factor,
                                        (y1 - self.offset_y) / self.zoom_factor,
                                        (x2 - self.offset_x) / self.zoom_factor,
                                        (y2 - self.offset_y) / self.zoom_factor,
                                        color))
                self.rect_table.insert("", "end", values=(x1, y1, x2, y2, color))
            else:
                self.label_image.delete(self.current_rectangle)
            self.current_rectangle = None

    def zoom_image(self, event):
        zoom_amount = 1.1 if event.delta > 0 else 0.9
        self.zoom_factor *= zoom_amount
        self.offset_x = event.x - zoom_amount * (event.x - self.offset_x)
        self.offset_y = event.y - zoom_amount * (event.y - self.offset_y)
        self.display_image()

    def delete_rectangle(self):
        selected_item = self.rect_table.selection()
        if selected_item:
            item = selected_item[0]
            index = self.rect_table.index(item)
            self.rect_table.delete(item)
            self.rectangles.pop(index)
            self.display_image()

    def on_resize(self, event):
        if self.spec_image is not None:
            self.display_image()

    def choose_color(self):
        return colorchooser.askcolor(title="Choose Rectangle Color")[1]


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('1200x800')
    app = App(root)
    root.mainloop()
