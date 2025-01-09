import tkinter as tk
from tkinter import filedialog, colorchooser, ttk
import spectral
import numpy as np
from PIL import Image, ImageTk
# import pandas as pd
# import cv2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import matplotlib.pyplot as plt


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

        # Bind the resize event to the canvas
        self.label_image.bind("<Configure>", self.on_resize)

        # Right label with parameters
        self.right_label = tk.Label(self.frame)
        self.right_label.config(justify='left', background='white', width=40)
        self.right_label.grid(row=1, column=1, sticky='ns')

        self.rect_table = ttk.Treeview(self.right_label, columns=("x1", "y1", "x2", "y2", "color", "data"), show="headings")
        self.rect_table.heading("x1", text="X1")
        self.rect_table.heading("y1", text="Y1")
        self.rect_table.heading("x2", text="X2")
        self.rect_table.heading("y2", text="Y2")
        self.rect_table.heading("color", text="Color")
        self.rect_table.heading("data", text="Data")
        self.rect_table.column("x1", width=50)
        self.rect_table.column("y1", width=50)
        self.rect_table.column("x2", width=50)
        self.rect_table.column("y2", width=50)
        self.rect_table.column("color", width=50)
        self.rect_table.column("data", width=40)
        self.rect_table.grid(row=0, column=0, sticky='nw')

        # Buttons for table actions
        self.delete_button = tk.Button(self.right_label, text="Delete Selected", command=self.delete_rectangle)
        self.delete_button.grid(row=1, column=0, sticky="nwe")

        # self.edit_button = tk.Button(self.right_label, text="Edit Selected", command=self.edit_rectangle)
        # self.edit_button.pack(fill=tk.X)

        # Variables
        self.spec_image = None
        self.spec_image_info = None
        self.rgb_image = None  # Initialize here to avoid AttributeError
        self.start_x = None
        self.start_y = None
        self.rectangles = []  # Stores rectangles as (x1, y1, x2, y2, color)
        self.current_rectangle = None

        # Rectangle
        self.label_image.bind("<ButtonPress-1>", self.start_draw)
        self.label_image.bind("<B1-Motion>", self.draw_rectangle)
        self.label_image.bind("<ButtonRelease-1>", self.complete_rectangle)

    def open_file(self):
        """Handles file opening logic."""
        file_path = filedialog.askopenfilename(title='Open File',
                                               filetypes=(("Spectra File", "*.hdr"), ("All files", "*.*")))

        if file_path.endswith('.hdr'):
            # Load the image and metadata
            self.spec_image = spectral.open_image(file_path).load()

            # Extract the metadata dictionary
            metadata = self.spec_image.metadata

            default_bands = metadata.get("default bands", [])
            if default_bands:
                default_bands = [int(band) for band in default_bands]

            # Extract relevant metadata fields
            self.spec_image_info = {
                "samples": int(metadata.get("samples", 0)),
                "bands": int(metadata.get("bands", 0)),
                "lines": int(metadata.get("lines", 0)),
                "data type": int(metadata.get("data type", "Unknown")),
                "default bands": default_bands,
                "wavelengths": metadata.get("wavelength", []),
            }

            if self.spec_image_info is None:
                return

            bands = self.spec_image_info['default bands']
            rgb_image = np.zeros((self.spec_image_info["lines"], self.spec_image_info["samples"], 3), dtype=np.float32)
            for i, band_index in enumerate(bands):
                band_data = np.squeeze(self.spec_image[:, :, band_index])  # Extract and squeeze band data to ensure 2D shape
                max_value = np.amax(band_data)  # Get the maximum value for normalization

                # Avoid division by zero
                if max_value > 0:
                    rgb_image[:, :, i] = band_data / max_value

            # Scale to 0-255 range and convert to uint8 for display
            self.rgb_image = (rgb_image * 255).astype(np.uint8)
            self.display_image()

    def display_image(self):
        """Resize and display the image with scaled rectangles."""
        if self.rgb_image is None:
            return

        pil_image = Image.fromarray(self.rgb_image)

        # Get canvas size
        canvas_width = self.label_image.winfo_width()
        canvas_height = self.label_image.winfo_height()

        # Calculate aspect ratio
        img_width, img_height = pil_image.size
        aspect_ratio = img_width / img_height

        # Resize image to fit the canvas
        new_width = canvas_width
        new_height = int(new_width / aspect_ratio)

        if new_height > canvas_height:
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)

        # Save the scale factors
        self.scale_x = new_width / img_width
        self.scale_y = new_height / img_height

        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.label_image.delete("all")  # Clear canvas before drawing new image
        self.label_image.create_image(0, 0, anchor="nw", image=self.tk_image)

        # Redraw rectangles with new coordinates
        for orig_x1, orig_y1, orig_x2, orig_y2, color in self.rectangles:
            scaled_x1 = orig_x1 * self.scale_x
            scaled_y1 = orig_y1 * self.scale_y
            scaled_x2 = orig_x2 * self.scale_x
            scaled_y2 = orig_y2 * self.scale_y
            self.label_image.create_rectangle(scaled_x1, scaled_y1, scaled_x2, scaled_y2, outline=color, width=2)

    def start_draw(self, event):
        """Start drawing a rectangle."""
        self.start_x = event.x
        self.start_y = event.y
        self.current_rectangle = self.label_image.create_rectangle(self.start_x, self.start_y, event.x, event.y,
                                                                   outline="red", width=2)

    def draw_rectangle(self, event):
        """Update the rectangle as the user drags the mouse."""
        if self.current_rectangle:
            self.label_image.coords(self.current_rectangle, self.start_x, self.start_y, event.x, event.y)

    def complete_rectangle(self, event):
        """Complete the rectangle and allow the user to choose a color."""
        if self.current_rectangle:
            x1, y1, x2, y2 = self.label_image.coords(self.current_rectangle)

            # Convert canvas coordinates to original image coordinates
            orig_x1 = x1 / self.scale_x
            orig_y1 = y1 / self.scale_y
            orig_x2 = x2 / self.scale_x
            orig_y2 = y2 / self.scale_y

            color = self.choose_color()
            if color:
                self.label_image.itemconfig(self.current_rectangle, outline=color)
                self.rectangles.append((orig_x1, orig_y1, orig_x2, orig_y2, color))  # Store in original space
                self.rect_table.insert("", "end", values=(orig_x1, orig_y1, orig_x2, orig_y2, color))
            else:
                self.label_image.delete(self.current_rectangle)
            self.current_rectangle = None

    def choose_color(self):
        """Allow the user to choose a color for the rectangle."""
        color = colorchooser.askcolor(title="Choose Rectangle Color")[1]
        return color

    def delete_rectangle(self):
        """Delete the selected rectangle from the canvas and table."""
        selected_item = self.rect_table.selection()
        if selected_item:
            item = selected_item[0]  # Get the selected item in the table
            index = self.rect_table.index(item)  # Get the index of the selected item
            self.rect_table.delete(item)  # Remove the item from the table

            # Remove the corresponding rectangle from the canvas
            rect_to_delete = self.rectangles.pop(index)  # Remove from the rectangles list
            x1, y1, x2, y2, color = rect_to_delete
            for rect_id in self.label_image.find_all():  # Find all items on the canvas
                coords = self.label_image.coords(rect_id)  # Get the coordinates of the item
                if coords == [x1, y1, x2, y2]:  # Match coordinates
                    self.label_image.delete(rect_id)  # Delete the canvas rectangle
                    break

    def edit_rectangle(self):
        """Edit the color of the selected rectangle."""
        selected_item = self.rect_table.selection()
        if selected_item:
            item = selected_item[0]  # Get the selected item in the table
            index = self.rect_table.index(item)  # Get the index of the selected item
            new_color = self.choose_color()  # Let the user choose a new color

            if new_color:
                x1, y1, x2, y2, _ = self.rectangles[index]  # Get the rectangle data
                self.rectangles[index] = (x1, y1, x2, y2, new_color)  # Update the list

                # Update the table entry
                self.rect_table.item(item, values=(x1, y1, x2, y2, new_color))

                # Update the canvas rectangle color
                for rect_id in self.label_image.find_all():  # Find all items on the canvas
                    coords = self.label_image.coords(rect_id)  # Get the coordinates of the item
                    if coords == [x1, y1, x2, y2]:  # Match coordinates
                        self.label_image.itemconfig(rect_id, outline=new_color)  # Change the outline color
                        break

    def on_resize(self, event):
        """Handle canvas resize events."""
        if self.rgb_image is not None:
            self.display_image()



if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('1200x800')
    app = App(root)
    root.mainloop()