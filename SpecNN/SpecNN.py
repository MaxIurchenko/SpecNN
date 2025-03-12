import tkinter as tk
from tkinter import filedialog, colorchooser, ttk, messagebox

import spectral
import numpy as np
from PIL import Image, ImageTk, ImageGrab
from keras.src.utils.module_utils import tensorflow
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Alloy Spectre Detection with Neural Networks")

        # Create the main frame
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.frame.grid_rowconfigure(0, weight=0)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_rowconfigure(2, weight=0)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=0)

        # Top menu -----------------------------------------------------------------------------------------------
        self.top_menu = tk.Label(self.frame)
        self.top_menu.config(justify='left', background="lightgray", height=2)
        self.top_menu.grid(row=0, column=0, columnspan=2, sticky='nwe')

        self.open_file_button = tk.Button(self.top_menu, text="Open HDR", command=self.open_file)
        self.open_file_button.pack(side="left", padx=0, pady=0)

        # Bind the resize event and mouse motion
        self.toggle_button = tk.Button(self.top_menu, text="Magnifying Glass", command=self.toggle_magnifier)
        self.toggle_button.pack(pady=0, side='left')

        # Clear RGB image
        self.toggle_button = tk.Button(self.top_menu, text="Clear RGB", command=self.clear_rgb_image)
        self.toggle_button.pack(pady=0, side='left')

        # Increase brightness RGB image
        self.toggle_button = tk.Button(self.top_menu, text="Increase brightness", command=self.increase_brightness)
        self.toggle_button.pack(pady=0, side='left')


        # Save RGB image
        self.toggle_button = tk.Button(self.top_menu, text="Save RGB", command=self.save_rgb_image)
        self.toggle_button.pack(pady=0, side='left')

        # Image display-----------------------------------------------------------------------------------------------
        self.label_image = tk.Canvas(self.frame, bg="white")
        self.label_image.grid(row=1, column=0, sticky="news")

        # Bind the resize event to the canvas
        self.label_image.bind("<Configure>", self.on_resize)

        # Create a text widget to display the model summary
        self.summary_label = tk.Label(self.frame, font=("Arial", 12, "bold"), height=10)
        self.summary_label.grid(row=2, column=0, sticky="sew", padx=0, pady=0)
        # Create a text box to hold the model summary
        self.summary_text = tk.Text(self.summary_label, wrap="word", font=("Tahoma", 10), width=250, height=10)
        self.summary_text.grid(sticky="nsew", padx=0, pady=0)

        # Right label with parameters
        self.right_menu_label = tk.Canvas(self.frame, width=285, height=600, background='white')
        self.scroll_right_label = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.right_menu_label.yview)

        self.scroll_right_label.grid(row=1, column=2,rowspan=2, sticky="nse")
        self.right_menu_label.grid(row=1, rowspan=2, column=1, sticky="nsew")
        self.right_menu_label.config(yscrollcommand=self.scroll_right_label.set)

        # **Frame inside Canvas**
        self.right_label = tk.Frame(self.right_menu_label, background="white")
        self.window_id = self.right_menu_label.create_window((0, 0), window=self.right_label, anchor="nw")

        # **Function to update scroll region dynamically**
        def update_scroll_region(event):
            self.right_menu_label.configure(scrollregion=self.right_menu_label.bbox("all"))

        self.right_label.bind("<Configure>", update_scroll_region)

        # RGB combobox
        self.red_combo_box = ttk.Combobox(self.right_label, state='readonly')
        self.green_combo_box = ttk.Combobox(self.right_label, state='readonly')
        self.blue_combo_box = ttk.Combobox(self.right_label, state='readonly')
        self.update_bands_button = tk.Button(self.right_label, text='Update\nbands', command=self.rgb_combobox, width=5,
                                             height=4)
        self.update_bands_button.grid(row=0, columnspan=2, column=0, sticky='w')
        self.red_combo_box.grid(row=0, columnspan=2, column=0, sticky='ne')
        self.green_combo_box.grid(row=0, columnspan=2, column=0, sticky='e')
        self.blue_combo_box.grid(row=0, columnspan=2, column=0, sticky='se')


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
        self.rect_table.grid(row=1, column=0, sticky='nw')

        # Bind this method to the treeview
        self.rect_table.bind("<Double-1>", self.edit_data_cell)

        # Buttons for table actions
        self.delete_button = tk.Button(self.right_label, text="Delete Selected", command=self.delete_rectangle)
        self.delete_button.grid(row=2, column=0, sticky="nwe")

        # Button save preparing data MLP
        self.delete_button = tk.Button(self.right_label, text="Save train data MLP", command=self.save_train_data)
        self.delete_button.grid(row=3, column=0, sticky="nwe")

        self.patch = tk.Label(self.right_label, text="Patch size", font=("Arial", 12))
        self.patch.grid(row=4, column=0, sticky="w", padx=0, pady=0)
        self.patch_value = ttk.Entry(self.right_label, font=("Arial", 12))
        self.patch_value.grid(row=4, column=0, sticky="nse", padx=0, pady=0)
        self.patch_value.delete(0, tk.END)  # Clear any existing value
        self.patch_value.insert(0, str(3))  # Insert the provided default value as a string
        self.patch_value.config(validate="key", validatecommand=(self.root.register(self.validate_int), "%P"))

        # Button save preparing data CNN
        self.delete_button = tk.Button(self.right_label, text="Save train data CNN", command=self.save_train_data_cnn)
        self.delete_button.grid(row=5, column=0, sticky="nwe")

        # Button save preparing data to file
        self.delete_button = tk.Button(self.right_label, text="Save train data to file", command=self.save_train_data_to_file)
        self.delete_button.grid(row=6, column=0, sticky="nwe")

        # Button load preparing data to file
        self.delete_button = tk.Button(self.right_label, text="Load train data", command=self.load_train_data)
        self.delete_button.grid(row=7, column=0, sticky="nwe")

        # Button clear preparing data to file
        self.delete_button = tk.Button(self.right_label, text="Clear train data", command=self.clear_train_data)
        self.delete_button.grid(row=8, column=0, sticky="nwe")

        # Label to display train_x and train_y shapes
        self.train_shape_label = tk.Label(self.right_label, text="train_x: N/A, train_y: N/A", font=("Arial", 12))
        self.train_shape_label.grid(row=9, column=0, sticky="nwe")

        # Variables
        self.spec_image = None
        self.spec_image_info = None
        self.rgb_image = None  # Initialize here to avoid AttributeError
        self.start_x = None
        self.start_y = None
        self.rectangles = []  # Stores rectangles as (x1, y1, x2, y2, color)
        self.current_rectangle = None
        self.scale_x = 1
        self.scale_y = 1
        self.magnifier_window = None
        self.is_magnifier_active = False
        self.stop_training_callback = StopTrainingCallback()
        self.test_x, self.test_y = None, None
        self.batch_size_value = None
        self.epoch_value = None
        self.max_value = None
        self.min_value = None
        self.dataset = None
        self.patch_size = 5
        self.accuracy = 0.5

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

            # Extract or compute wavelengths
            wavelengths = metadata.get("wavelength", [])
            if isinstance(wavelengths, list) and len(wavelengths) == 1 and isinstance(wavelengths[0], str):
                wavelengths = [float(w) for w in wavelengths[0].splitlines()]
            elif isinstance(wavelengths, list):
                wavelengths = [float(w) for w in wavelengths]

            metadata["wavelengths"] = wavelengths

            # Determine default bands
            target_value = 430
            val1 = min(enumerate(metadata["wavelengths"]), key=lambda x: abs(x[1] - target_value))
            val1_idx = val1[0]  # Extract the index

            val2 = min(enumerate(metadata["wavelengths"]),
                       key=lambda x: abs(x[1] - (metadata["wavelengths"][val1_idx] + 100)))
            val2_idx = val2[0]  # Extract the index

            val3 = min(enumerate(metadata["wavelengths"]),
                       key=lambda x: abs(x[1] - (metadata["wavelengths"][val1_idx] + 200)))
            val3_idx = val3[0]  # Extract the index

            default_bands = [val1, val2, val3]
            metadata["default bands"] = [default_bands[0][0],default_bands[1][0],default_bands[2][0]]

            default_bands = metadata.get("default bands", [])
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

            # Get the max and min values of the spectral image
            self.max_value = np.amax(self.spec_image)
            self.min_value = np.amin(self.spec_image)


            bands = self.spec_image_info['default bands']
            print(self.spec_image_info['default bands'])
            rgb_image = np.zeros((self.spec_image_info["lines"], self.spec_image_info["samples"], 3), dtype=np.float32)
            for i, band_index in enumerate(bands):
                band_data = np.squeeze(self.spec_image[:, :, band_index])  # Extract and squeeze band data to ensure 2D shape
                max_value = np.amax(band_data)  # Get the maximum value for normalization

                # Avoid division by zero

                rgb_image[:, :, i] = band_data / max_value

            self.red_combo_box.config(values=self.spec_image_info['wavelengths'])
            self.green_combo_box.config(values=self.spec_image_info['wavelengths'])
            self.blue_combo_box.config(values=self.spec_image_info['wavelengths'])
            self.red_combo_box.current(self.spec_image_info['default bands'][0])
            self.green_combo_box.current(self.spec_image_info['default bands'][1])
            self.blue_combo_box.current(self.spec_image_info['default bands'][2])

            # Scale to 0-255 range and convert to uint8 for display

            self.rgb_image = (rgb_image * 255).astype(np.uint8)
            self.display_image()

    def rgb_combobox(self):

        # bands = self.spec_image_info['default bands']
        bands = [self.red_combo_box.current(), self.green_combo_box.current(), self.blue_combo_box.current()]
        # print(self.spec_image_info['default bands'])
        rgb_image = np.zeros((self.spec_image_info["lines"], self.spec_image_info["samples"], 3), dtype=np.float32)
        for i, band_index in enumerate(bands):
            band_data = np.squeeze(
                self.spec_image[:, :, band_index])  # Extract and squeeze band data to ensure 2D shape
            max_value = np.amax(band_data)  # Get the maximum value for normalization

            # Avoid division by zero

            rgb_image[:, :, i] = band_data / max_value

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
        for orig_x1, orig_y1, orig_x2, orig_y2, color, data in self.rectangles:
            scaled_x1 = orig_x1 * self.scale_x
            scaled_y1 = orig_y1 * self.scale_y
            scaled_x2 = orig_x2 * self.scale_x
            scaled_y2 = orig_y2 * self.scale_y
            self.label_image.create_rectangle(scaled_x1, scaled_y1, scaled_x2, scaled_y2, outline=color, width=2)

    def save_rgb_image(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")],
            title="Save RGB Image As"
        )
        if not file_path:  # User canceled the save dialog
            print("Save operation canceled.")
            return
        if self.rgb_image is None:
            print("No spectral image loaded!")
            return

        # # Extract the selected bands (assuming bands are in [0, num_bands-1] range)
        # rgb_image = self.spec_image[:, :, bands]
        #
        # # Normalize to 0-255 for visualization
        # rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255
        # rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        # Convert to PIL image and save
        img = Image.fromarray(self.rgb_image)
        img.save(file_path)
        print(f"RGB image saved as {file_path}")
        
    def increase_brightness(self):
        self.rgb_image *= 2
        self.display_image()

    def clear_rgb_image(self):
        bands = self.spec_image_info['default bands']
        rgb_image = np.zeros((self.spec_image_info["lines"], self.spec_image_info["samples"], 3), dtype=np.float32)
        for i, band_index in enumerate(bands):
            band_data = np.squeeze(
                self.spec_image[:, :, band_index])  # Extract and squeeze band data to ensure 2D shape
            max_value = np.amax(band_data)  # Get the maximum value for normalization

            # Avoid division by zero
            if max_value > 0:
                rgb_image[:, :, i] = band_data / max_value

        # Scale to 0-255 range and convert to uint8 for display
        self.rgb_image = (rgb_image * 255).astype(np.uint8)
        self.display_image()

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
            orig_x1 = int(x1 / self.scale_x)
            orig_y1 = int(y1 / self.scale_y)
            orig_x2 = int(x2 / self.scale_x)
            orig_y2 = int(y2 / self.scale_y)

            color = self.choose_color()
            if color:
                self.label_image.itemconfig(self.current_rectangle, outline=color)
                self.rectangles.append((orig_x1, orig_y1, orig_x2, orig_y2, color, 0))  # Store in original space

                # Create a tag for this color
                tag_name = f"color_{color.replace('#', '')}"  # Unique tag based on color
                self.rect_table.tag_configure(tag_name, background=color)  # Set background color

                # Insert the row with the tag
                self.rect_table.insert("", "end", values=(orig_x1, orig_y1, orig_x2, orig_y2, color, 0),
                                       tags=(tag_name,))
            else:
                self.label_image.delete(self.current_rectangle)
            self.current_rectangle = None

    def set_treeview_cell_color(self, treeview, iid: str, colnum: int, bg_color):
        """
        Set the color of a specific Treeview cell using a canvas overlay.
        """

        def _apply_canvas():
            # Get the text of the cell
            cell_value = treeview.item(iid, "values")[colnum]

            # Get the text anchor of the column
            x_padding = 4
            anchor = treeview.column(treeview["columns"][colnum], "anchor")

            # Get cell location, width, and height
            bbox = treeview.bbox(iid, colnum)
            if not bbox:
                return  # Skip if bbox is not available
            x, y, width, height = bbox

            # Create the canvas
            canvas = tk.Canvas(
                master=treeview, background=bg_color, borderwidth=0, highlightthickness=0
            )
            canvas.configure(width=width, height=height)

            # Add the canvas to the Treeview
            canvas.place(in_=treeview, x=x, y=y)

            # Save reference to canvas for management
            if not hasattr(treeview, "canvases"):
                treeview.canvases = []
            treeview.canvases.append(canvas)

        # Ensure Treeview is visible before applying canvas
        treeview.after(100, _apply_canvas)

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

            # Remove the corresponding rectangle from the list of rectangles
            rect_to_delete = self.rectangles.pop(index)  # Remove from the rectangles list
            x1, y1, x2, y2, color, data = rect_to_delete

            # Find and delete the corresponding rectangle on the canvas
            for rect_id in self.label_image.find_all():  # Find all items on the canvas
                scaled_coords = [x1 * self.scale_x, y1 * self.scale_y, x2 * self.scale_x, y2 * self.scale_y]
                coords = self.label_image.coords(rect_id)  # Get the coordinates of the item
                if (abs(coords[0] - scaled_coords[0]) < 1 and
                        abs(coords[1] - scaled_coords[1]) < 1 and
                        abs(coords[2] - scaled_coords[2]) < 1 and
                        abs(coords[3] - scaled_coords[3]) < 1):  # Match coordinates
                    self.label_image.delete(rect_id)  # Delete the canvas rectangle
                    break

            # Remove the canvas overlay for the cell background color
            if hasattr(self.rect_table, "canvases"):
                # Match the canvas to the index
                if index < len(self.rect_table.canvases):  # Ensure index is valid
                    canvas = self.rect_table.canvases.pop(index)
                    canvas.destroy()  # Remove the canvas from the Treeview

            # Re-index canvases for remaining rows, if necessary
            if hasattr(self.rect_table, "canvases"):
                for idx, canvas in enumerate(self.rect_table.canvases):
                    canvas_coords = self.rect_table.bbox(self.rect_table.get_children()[idx], column=4)
                    if canvas_coords:
                        x, y, width, height = canvas_coords
                        canvas.place(x=x, y=y, width=width, height=height)

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

    def toggle_magnifier(self):
        """Toggle the magnifier on or off."""
        if self.is_magnifier_active:
            self.close_magnifier()
        else:
            self.open_magnifier()

    def open_magnifier(self):
        """Open the magnifying glass as a borderless window."""
        self.is_magnifier_active = True
        self.magnifier_window = tk.Toplevel(self.root)
        self.magnifier_window.overrideredirect(True)  # Remove borders
        self.magnifier_window.attributes("-topmost", True)  # Keep on top
        self.magnifier_window.geometry("150x150")  # Initial size
        self.magnifier_window.configure(bg="black")

        # Canvas to display magnified content
        self.magnifier_canvas = tk.Canvas(self.magnifier_window, width=150, height=150, bg="black",
                                          highlightthickness=0)
        self.magnifier_canvas.pack(fill=tk.BOTH, expand=True)

        # Start the magnification loop
        self.update_magnifier()

    def close_magnifier(self):
        """Close the magnifying glass."""
        self.is_magnifier_active = False
        if self.magnifier_window:
            self.magnifier_window.destroy()
            self.magnifier_window = None

    def update_magnifier(self):
        """Update the magnified view."""
        if not self.is_magnifier_active:
            return

        # Get mouse position
        x, y = self.root.winfo_pointerx(), self.root.winfo_pointery()

        # Define magnification area (centered around the cursor)
        magnifier_size = 50  # Half-size of the area to magnify
        x1 = x - magnifier_size
        y1 = y - magnifier_size
        x2 = x + magnifier_size
        y2 = y + magnifier_size

        # Capture screen region

        screen = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        magnified_image = screen.resize((150, 150), Image.Resampling.NEAREST)
        self.magnifier_window.deiconify()

        # Display on magnifier canvas
        self.tk_magnified = ImageTk.PhotoImage(magnified_image)

        self.magnifier_canvas.delete("all")
        self.magnifier_canvas.create_image(0, 0, anchor="nw", image=self.tk_magnified)

        # Position magnifier near the cursor
        self.magnifier_window.geometry(f"150x150+{x + 55}+{y}")
        self.magnifier_canvas.create_line(60, 75, 90, 75, fill="red",
                                          width=2)  # Diagonal line from top-left to bottom-right
        self.magnifier_canvas.create_line(75, 60, 75, 90, fill="red",
                                          width=2)  # Diagonal line from bottom-left to top-right

        # Schedule the next update
        self.root.after(30, self.update_magnifier)

    def on_resize(self, event):
        """Handle canvas resize events."""
        if self.rgb_image is not None:
            self.display_image()

    def edit_data_cell(self, event):
        """Enable editing for the 'data' field in the table."""
        selected_item = self.rect_table.selection()
        if not selected_item:
            return

        item_id = selected_item[0]
        columns = self.rect_table["columns"]
        data_column_index = columns.index("data") if "data" in columns else None

        if data_column_index is None:
            print("Error: 'data' column not found")
            return

        values = self.rect_table.item(item_id, "values")
        if data_column_index >= len(values):
            print(f"Error: 'data' column index {data_column_index} out of range for values {values}")
            return

        # Get the current value
        current_value = values[data_column_index]

        # Create an entry widget for editing
        x, y, width, height = self.rect_table.bbox(item_id, f"#{data_column_index + 1}")
        entry = tk.Entry(self.rect_table, justify="center")
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus()
        entry.insert(0, current_value)

        def save_edit(event=None):
            try:
                new_value = int(entry.get())  # Ensure valid integer input
            except ValueError:
                new_value = current_value  # Revert to the old value if invalid
            updated_values = list(values)
            updated_values[data_column_index] = new_value  # Update the data column
            self.rect_table.item(item_id, values=updated_values)  # Update the table

            # Update the corresponding rectangle in self.rectangles
            index = self.rect_table.index(item_id)  # Find the index of the edited row
            if index < len(self.rectangles):  # Ensure index is within bounds
                rect = list(self.rectangles[index])
                rect[5] = new_value  # Assuming column 5 corresponds to the data value
                self.rectangles[index] = tuple(rect)  # Update the rectangle with the new value

            entry.destroy()

        # Bind save on Enter or focus out
        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", save_edit)

    def update_shape_label(self):
        """Update the label displaying the shapes of train_x and train_y."""
        if self.train_x is not None and self.train_y is not None:
            text = f"train_x: {self.train_x.shape}, train_y: {self.train_y.shape}"
        else:
            text = "train_x: N/A, train_y: N/A"
        self.train_shape_label.config(text=text)

    def save_train_data(self):
        """Generate training data from the rectangles and spec_image."""
        self.train_x = []
        self.train_y = []

        if self.spec_image is None:
            print("No spectral image loaded!")
            return

        for (x1, y1, x2, y2, color, data) in self.rectangles:
            # Extract spectral data from the spec_image within the rectangle (x1, y1, x2, y2)
            spectral_data = np.array(self.spec_image[y1:y2, x1:x2, :])  # (height, width, num_bands)
            print(f"specral_data.shape: {spectral_data.shape}")

            # Flatten spectral data: (height, width, num_bands) → (num_pixels, num_bands)
            flattened_spectral_data = spectral_data.reshape(-1, spectral_data.shape[2])

            # Append correctly to lists
            self.train_x.extend(flattened_spectral_data)  # Properly appends each pixel's spectral values

            # If data == 0, assign zero label
            if data == 0:
                self.train_y.extend([0] * flattened_spectral_data.shape[0])  # Label as 0
            else:
                self.train_y.extend([data] * flattened_spectral_data.shape[0])  # Normal label

        # Convert lists to numpy arrays
        self.train_x = np.array(self.train_x)  # Shape: (total_pixels, num_bands)
        self.train_y = np.array(self.train_y)  # Shape: (total_pixels,)

        # Normalize train_x
        range_value = self.max_value - self.min_value
        self.train_x = (self.train_x - self.min_value) / (range_value + 1e-10)  # Avoid division by zero

        print(f"train_x shape: {self.train_x.shape}")
        print(f"train_y shape: {self.train_y.shape}")

        # Encode labels
        unique_values, self.counts = np.unique(self.train_y, return_counts=True)
        self.output_neurons = len(unique_values)

        # Handle one-hot encoding
        label_encoder = LabelEncoder()
        self.train_y = label_encoder.fit_transform(self.train_y)  # Encode labels
        self.train_y = keras.utils.to_categorical(self.train_y, num_classes=self.output_neurons)  # One-hot encode

        # If `data == 0`, ensure labels are all-zero vectors
        if 0 in label_encoder.classes_:
            zero_index = np.where(label_encoder.classes_ == 0)[0][0]  # Find index of 0 in labels
            for i in range(len(self.train_y)):
                if np.argmax(self.train_y[i]) == zero_index:
                    self.train_y[i] = np.zeros(self.output_neurons)  # Replace with (0,0,0,...)

        print(f"MLP data prepared: {self.train_x.shape}, {self.train_y.shape}")

        self.update_shape_label()
        self.create_simple_mlp_parameters()

    def save_train_data_cnn(self):
        """Generate training data for CNN from the rectangles and spec_image."""
        self.train_x = []
        self.train_y = []

        if self.spec_image is None:
            print("No spectral image loaded!")
            return

        self.patch_size = int(self.patch_value.get())  # CNN requires (3,3) patches
        pad_size = self.patch_size // 2  # Padding size to extract 3×3 patches

        # Pad the spectral image to handle edge cases
        padded_spec_image = np.pad(self.spec_image,
                                   ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                                   mode='reflect')

        for (x1, y1, x2, y2, color, data) in self.rectangles:
            for i in range(y1, y2):
                for j in range(x1, x2):
                    # Extract 3x3 patch centered at (i, j)
                    patch = padded_spec_image[i:i + self.patch_size, j:j + self.patch_size, :]

                    if patch.shape == (self.patch_size, self.patch_size, self.spec_image.shape[2]):  # Ensure correct shape
                        self.train_x.append(patch)
                        self.train_y.append(data)

        # Convert lists to numpy arrays
        self.train_x = np.array(self.train_x)  # Shape: (num_samples, 3, 3, num_bands)
        self.train_y = np.array(self.train_y)  # Shape: (num_samples,)

        # Check if data exists
        if len(self.train_x) == 0 or len(self.train_y) == 0:
            print("Error: No valid training samples found! Please check input data.")
            return

        # **Remove `data == 0` samples only if 0 is unclassified**
        valid_indices = np.where(self.train_y != 0)[0]
        if len(valid_indices) == 0:
            print("Error: All samples were removed during filtering (data == 0).")
            return

        self.train_x = self.train_x[valid_indices]
        self.train_y = self.train_y[valid_indices]

        # **Normalize `train_x`**
        range_value = self.max_value - self.min_value
        if range_value == 0:
            print("Warning: max_value and min_value are the same, normalization will fail!")
            range_value = 1  # Avoid division by zero

        self.train_x = (self.train_x - self.min_value) / (range_value + 1e-10)

        # **Encode labels properly**
        label_encoder = LabelEncoder()
        self.train_y = label_encoder.fit_transform(self.train_y)
        self.output_neurons = len(label_encoder.classes_)

        # Ensure output neurons are valid before one-hot encoding
        if self.output_neurons == 0:
            print("Error: No unique labels found after encoding!")
            return

        self.train_y = to_categorical(self.train_y, num_classes=self.output_neurons)

        print(f"CNN data prepared: {self.train_x.shape}, {self.train_y.shape}")

        self.update_shape_label()
        self.create_cnn_parameters()  # Call CNN-specific model setup

    def save_train_data_to_file(self):
        """Save the generated training data to a .npz file."""
        if not hasattr(self, 'train_x') or not hasattr(self, 'train_y'):
            print("Training data not generated yet!")
            return

        # Save train_x and train_y to a .npz file
        file_path = filedialog.asksaveasfilename(
            title='Save File',
            filetypes=(("Numpy Spectra File", "*.npz"), ("All files", "*.*")),
            defaultextension=".npz"
        )

        if file_path:  # Ensure the user didn't cancel the save dialog
            np.savez(file_path, train_x=self.train_x, train_y=self.train_y)
            print(f"Training data saved to '{file_path}'")
        else:
            print("Save operation was canceled.")

    def load_train_data(self):
        """Load training data from a .npz file."""
        # Open a file dialog for the user to select the file
        file_path = filedialog.askopenfilename(
            title='Select Training Data File',
            filetypes=(("NPZ Files", "*.npz"), ("All Files", "*.*"))
        )

        # Check if a file was selected
        if not file_path:
            print("No file selected.")
            return

        try:
            # Load the .npz file
            data = np.load(file_path)

            # Extract train_x and train_y from the file
            train_x = data['train_x']
            train_y = data['train_y']

            # If self.train_x exists, append; otherwise, initialize
            if hasattr(self, 'train_x') and hasattr(self, 'train_y'):
                self.train_x = np.vstack((self.train_x, train_x))
                self.train_y = np.hstack((self.train_y, train_y))
            else:
                self.train_x = train_x
                self.train_y = train_y

            self.update_shape_label()

            print("Training data loaded successfully.")
            print(f"train_x shape: {self.train_x.shape}, train_y shape: {self.train_y.shape}")

        except Exception as e:
            print(f"Error loading training data: {e}")

        self.update_shape_label()
        self.create_simple_mlp_parameters()

    def clear_train_data(self):
        """Clear the training data."""
        if hasattr(self, 'train_x') or hasattr(self, 'train_y'):
            self.train_x = None
            self.train_y = None
            print("Training data has been cleared.")
        else:
            print("No training data to clear.")
        self.update_shape_label()

    def hex_to_rgb(self,hex_color):
        hex_color = hex_color.lstrip("#")  # Remove '#' if present
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    #----------------------NeuralNetworks------------------------------------

    def create_simple_mlp_parameters(self):
        headers = ["Layer", "Configuration"]
        self.nn_parameters = tk.Label(self.right_label, font=("Arial", 12, "bold"))
        self.nn_parameters.grid(row=10, column=0, sticky="new")

        for col, header in enumerate(headers):
            label = tk.Label(self.nn_parameters, text=header, font=("Arial", 12, "bold"), bg="lightgray")
            label.grid(row=1, column=col, sticky="new", padx=2, pady=2)


        # Create the table rows
        self.create_table_row(2, "Input layer", f"{self.train_x.shape[1]}", readonly=True)
        self.first_hidden_layer = self.create_table_row(3, "First Hidden Layer", 64, input_type="int")
        self.second_hidden_layer = self.create_table_row(4, "Second Hidden Layer", 32, input_type="int")
        self.therd_hidden_layer = self.create_table_row(5, "Third Hidden Layer", 0, input_type="int")
        self.output_layer = self.create_table_row(6, "Output Layer", f"{self.output_neurons}", readonly=True)
        self.epoch = self.create_table_row(7, "Epoch", 20, input_type="int")
        self.batch_size = self.create_table_row(8, "Batch size", 32, input_type="int")
        self.dropout = self.create_table_row(9, "Dropout", 0.5, input_type="float")
        # self.kernel_regularizer = self.create_table_row(10, "Kernel regularizer", 0.001, input_type="float")

        # Add a button to print the configuration
        submit_btn = tk.Button(self.nn_parameters, text="Submit", command=self.make_nn_model, font=("Arial", 12))
        submit_btn.grid(row=11, column=0, columnspan=2, pady=10)

        self.start_train = tk.Button(self.right_label, text="Start train", command=self.start_train_nn)
        self.start_train.grid(row=14, column=0, sticky="new")
        self.test = tk.Button(self.right_label, text="Test", command=self.test_nn)
        self.test_accuracy = tk.Label(self.right_label, text="Test accuracy", font=("Arial", 12))
        self.test_accuracy.grid(row=15, column=0, sticky="w", padx=0, pady=0)
        self.test_accuracy_value = ttk.Entry(self.right_label, font=("Arial", 12))
        self.test_accuracy_value.grid(row=15, column=0, sticky="nse", padx=0, pady=0)
        self.test_accuracy_value.delete(0, tk.END)  # Clear any existing value
        self.test_accuracy_value.insert(0, str(0.5))  # Insert the provided default value as a string
        self.test_accuracy_value.config(validate="key", validatecommand=(self.root.register(self.validate_float), "%P"))

        self.test.grid(row=16, column=0, sticky="new")
        self.save = tk.Button(self.right_label, text="Save", command=self.save_nn)
        self.save.grid(row=17, column=0, sticky="new")

    def validate_int(self, value):
        """
        Validate that the input is an integer or empty (to allow corrections).
        :param value: The current value of the entry field.
        :return: True if valid, False otherwise.
        """
        return value.isdigit() or value == ""

    def validate_float(self, new_value):
        """Allow only valid float input (including empty string)."""
        if new_value == "":  # Allow empty input
            return True
        try:
            float(new_value)  # Check if it's a valid float
            return True
        except ValueError:
            return False

    def create_table_row(self, row, label_text, default_value, input_type=None, readonly=False):
        """Helper function to create a row in the table."""
        label = tk.Label(self.nn_parameters, text=label_text, font=("Arial", 12))
        label.grid(row=row, column=0, sticky="w", padx=0, pady=0)

        if readonly:
            # Create a label for readonly fields
            entry = tk.Label(self.nn_parameters, text=default_value, font=("Arial", 12), bg="white", relief="solid")
            entry.grid(row=row, column=1, sticky="nsew", padx=0, pady=0)
        else:
            # Create an entry for user input
            entry = ttk.Entry(self.nn_parameters, font=("Arial", 12))
            entry.grid(row=row, column=1, sticky="nsew", padx=0, pady=0)

            # Insert the actual default value
            entry.delete(0, tk.END)  # Clear any existing value
            entry.insert(0, str(default_value))  # Insert the provided default value as a string

            if input_type == "int":
                # Validate integer input
                entry.config(validate="key", validatecommand=(self.root.register(self.validate_int), "%P"))

        return entry

    def make_nn_model(self):

        self.batch_size_value = int(self.batch_size.get())
        self.epoch_value = int(self.epoch.get())

        # Ensure data is prepared for MLP
        input_shape = self.train_x.shape[1]
        first_hidden = int(self.first_hidden_layer.get())
        second_hidden = int(self.second_hidden_layer.get())
        third_hidden = int(self.therd_hidden_layer.get())
        dropout = float(self.dropout.get())
        # kernel_regularizer = float(self.kernel_regularizer.get())
        # print(dropout, kernel_regularizer)

        self.model = keras.Sequential()
        self.model.add(Dense(first_hidden, input_dim=input_shape, activation='relu'))
        if dropout > 0:
            self.model.add(Dropout(dropout))
        if second_hidden > 0:
            self.model.add(Dense(second_hidden, activation='relu'))
            if dropout > 0:
                self.model.add(Dropout(dropout))
        if third_hidden > 0:
            self.model.add(Dense(third_hidden, activation='relu'))
            if dropout > 0:
                self.model.add(Dropout(dropout))

        self.model.add(Dense(self.output_neurons, activation='sigmoid'))
        # self.model.add(Dense(self.output_neurons, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           # loss='sparse_categorical_crossentropy',
                           # loss='categorical_crossentropy',
                           metrics=['accuracy'])
        print("MLP Model Created")
        print(self.model.summary())

        # Clean up previous widgets and show model summary
        for widget in self.nn_parameters.winfo_children():
            widget.destroy()
        self.nn_parameters.destroy()

        self.show_model_summary()

        # Optionally, add a button to reset the UI
        self.reset_btn = tk.Button(self.right_label, text="Reset NN", command=self.reset_ui, font=("Arial", 12))
        self.reset_btn.grid(row=10, sticky="ew")

    def show_model_summary(self):
        """Display the model's summary after it's created."""
        # Clear previous content and insert new model summary
        self.summary_text.delete("1.0", tk.END)
        if self.model:
            from io import StringIO
            import sys

            buffer = StringIO()
            sys.stdout = buffer  # Redirect stdout to buffer
            self.model.summary()  # Print model summary
            sys.stdout = sys.__stdout__  # Reset stdout
            model_summary = buffer.getvalue()
            self.summary_text.insert(tk.END, model_summary)

    def create_cnn_parameters(self):

        # Create table headers
        headers = ["Layer", "Configuration"]
        self.nn_parameters = tk.Label(self.right_label, font=("Arial", 12, "bold"))
        self.nn_parameters.grid(row=10, column=0, sticky="new")

        for col, header in enumerate(headers):
            label = tk.Label(self.nn_parameters, text=header, font=("Arial", 12, "bold"), bg="lightgray")
            label.grid(row=1, column=col, sticky="new", padx=2, pady=2)

        # Create CNN parameter input fields
        self.create_table_row(2, "Input Shape", f"({self.patch_size},{self.patch_size},"
                                                f"{self.train_x.shape[-1]})", readonly=True)
        self.conv1_filters = self.create_table_row(3, "Conv Layer 1 Filters", 32, input_type="int")
        self.conv2_filters = self.create_table_row(4, "Conv Layer 2 Filters", 64, input_type="int")
        self.conv3_filters = self.create_table_row(5, "Conv Layer 3 Filters", 128, input_type="int")
        self.pool_size = self.create_table_row(6, "Pooling Size", 2, input_type="int")
        self.dense_units = self.create_table_row(7, "Dense Layer Units", 128, input_type="int")
        self.output_layer = self.create_table_row(8, "Output Layer", f"{self.output_neurons}", readonly=True)
        self.epochs = self.create_table_row(9, "Epochs", 20, input_type="int")
        self.batch_size = self.create_table_row(10, "Batch Size", 32, input_type="int")
        self.dropout = self.create_table_row(11, "Dropout", 0.5, input_type="float")
        # self.kernel_regularizer = self.create_table_row(12, "Kernel Regularizer", 0.001, input_type="float")

        # Submit button
        submit_btn = tk.Button(self.nn_parameters, text="Submit", command=self.make_cnn_model, font=("Arial", 12))
        submit_btn.grid(row=13, column=0, columnspan=2, pady=10)

        # Training buttons
        self.start_train = tk.Button(self.right_label, text="Start Train", command=self.start_train_cnn)
        self.start_train.grid(row=14, column=0, sticky="new")
        self.test_accuracy = tk.Label(self.right_label, text="Test accuracy", font=("Arial", 12))
        self.test_accuracy.grid(row=15, column=0, sticky="w", padx=0, pady=0)
        self.test_accuracy_value = ttk.Entry(self.right_label, font=("Arial", 12))
        self.test_accuracy_value.grid(row=15, column=0, sticky="nse", padx=0, pady=0)
        self.test_accuracy_value.delete(0, tk.END)  # Clear any existing value
        self.test_accuracy_value.insert(0, str(0.5))  # Insert the provided default value as a string
        self.test_accuracy_value.config(validate="key", validatecommand=(self.root.register(self.validate_float), "%P"))

        self.test = tk.Button(self.right_label, text="Test", command=self.test_cnn)
        self.test.grid(row=16, column=0, sticky="new")
        self.save = tk.Button(self.right_label, text="Save", command=self.save_cnn)
        self.save.grid(row=17, column=0, sticky="new")

    def make_cnn_model(self):
        """Create CNN model based on user-defined parameters."""

        self.batch_size_value = int(self.batch_size.get())
        self.epoch_value = int(self.epochs.get())

        # Extract values from UI
        input_shape = (self.patch_size, self.patch_size, self.train_x.shape[-1])  # (3,3,num_bands)
        conv1_filters = int(self.conv1_filters.get())
        conv2_filters = int(self.conv2_filters.get())
        conv3_filters = int(self.conv3_filters.get())
        pool_size = int(self.pool_size.get())
        dense_units = int(self.dense_units.get())
        dropout = float(self.dropout.get())
        # kernel_regularizer = float(self.kernel_regularizer.get())

        # Build CNN model
        self.model = Sequential()

        # First Conv Layer
        self.model.add(Conv2D(conv1_filters, kernel_size=(3, 3), activation='relu',
                              input_shape=input_shape, padding='same'))
        self.model.add(BatchNormalization()),
        if self.patch_size > 3:
            self.model.add(MaxPooling2D((pool_size, pool_size), padding='same'))

        # Second Conv Layer
        self.model.add(Conv2D(conv2_filters, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization()),
        self.model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        # Third Convolutional Block
        self.model.add(Conv2D(conv3_filters, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(GlobalAveragePooling2D())  # Reduce feature maps to a vector

        # Fully Connected Layers
        self.model.add(Dense(dense_units, activation='relu'))
        if dropout > 0:
            self.model.add(Dropout(dropout))
        self.model.add(Dense(int(dense_units/2), activation='relu'))
        if dropout > 0:
            self.model.add(Dropout(dropout))


        if self.output_neurons > 1:
            # Output Layer
            self.model.add(Dense(self.output_neurons, activation='softmax'))
            optimizer = Adam(learning_rate=1e-4)  # Reduce learning rate
            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        if self.output_neurons == 1:
            # If binary classification (2 classes: 0 or 1)
            self.model.add(Dense(1, activation='sigmoid'))  # Change activation
            optimizer = Adam(learning_rate=1e-4)  # Reduce learning rate
            self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        print("CNN Model Created")
        print(self.model.summary())

        # Show model summary in UI
        for widget in self.nn_parameters.winfo_children():
            widget.destroy()

        self.nn_parameters.destroy()
        self.show_model_summary()

        # Optionally, add a button to reset the UI
        self.reset_btn = tk.Button(self.right_label, text="Reset NN", command=self.reset_ui, font=("Arial", 12))
        self.reset_btn.grid(row=10, sticky="we")

    def reset_ui(self):
        """Reset the UI to the initial state for parameter entry."""
        self.reset_btn.destroy()
        self.start_train.destroy()
        self.test.destroy()
        self.save.destroy()
        self.test_accuracy_value.destroy()
        self.test_accuracy.destroy()

    def start_train_nn(self):
        """Start the training process with progress display."""

        if self.model is None:
            tk.messagebox.showerror("Error", "No model is initialized. Please build a model first.")
            return

        # Create a training progress callback that writes to summary_text
        progress_callback = TrainingProgressCallback(self.summary_text)

        self.train_x, val_x, self.train_y, val_y = train_test_split(self.train_x, self.train_y, test_size=0.2,
                                                                    random_state=42)

        # Train the model
        print(self.batch_size_value)
        print(self.epoch_value)
        self.stop_training_callback = StopTrainingCallback()  # Ensure the stop callback is available
        history = self.model.fit(
            self.train_x, self.train_y,
            batch_size=int(self.batch_size_value),
            epochs=int(self.epoch_value),
            validation_data=(val_x, val_y),
            callbacks=[self.stop_training_callback, progress_callback]
        )

        # Notify training completion
        tk.messagebox.showinfo("Training Complete", "The training process has been completed.")

    def test_nn(self):
        """Test the trained neural network on the entire spectral image with a progress bar."""
        if self.model is None:
            tk.messagebox.showerror("Error", "No model is trained yet. Train the model first.")
            return

        # Convert spectral image to NumPy array
        spectral_data = np.array(self.spec_image)  # Shape (height, width, num_bands)
        print(f"spectral_data.shape: {spectral_data.shape}")

        height, width, num_bands = spectral_data.shape

        # Flatten spectral data: (height, width, num_bands) → (height * width, num_bands)
        flattened_spectral_data = spectral_data.reshape(-1, num_bands)  # Shape: (total_pixels, num_bands)

        # Normalize the spectral data
        range_value = self.max_value - self.min_value
        flattened_spectral_data = (flattened_spectral_data - self.min_value) / (range_value + 1e-10)

        # Create a new window for progress tracking
        progress_window = tk.Toplevel(self.summary_label)
        progress_window.title("Testing Progress")

        # Progress bar
        progress_label = tk.Label(progress_window, text="Processing image...", font=("Arial", 10))
        progress_label.pack(pady=5)

        progress_bar = ttk.Progressbar(progress_window, length=400, mode='determinate')
        progress_bar.pack(pady=10)

        # Set progress bar range
        progress_bar["maximum"] = height  # Each row processed updates the bar

        # Predict the class for each pixel
        prediction = self.model.predict(flattened_spectral_data)  # Shape: (total_pixels, num_classes)
        print(f"Testing Results Shape: {prediction.shape}")
        print(f"Sample Predictions: {prediction[:5]}")

        # Ensure predictions match expected size
        expected_size = height * width
        if prediction.shape[0] != expected_size:
            print(f"Error: Expected {expected_size} predictions, but got {prediction.shape[0]}")
            return

        colors = np.array([self.hex_to_rgb(color) for (_, _, _, _, color, _) in self.rectangles])
        colors = colors[:self.output_neurons]  # Adjust based on the number of classes

        # Reshape prediction back to (height, width, num_classes)
        prediction = prediction.reshape(height, width, self.output_neurons)

        self.accuracy = float(self.test_accuracy_value.get())
        print(f"accuracy:{self.accuracy}")

        # Process each row and update progress bar
        for y in range(height):
            for x in range(width):
                # Find the predicted class with the highest probability
                max_value_index = np.argmax(prediction[y, x])
                max_value = prediction[y, x, max_value_index]

                # If confidence is high enough, assign color
                if max_value > self.accuracy:
                    self.rgb_image[y, x] = colors[max_value_index]

            # Update progress bar
            progress_bar["value"] = y
            progress_window.update_idletasks()  # Force UI update

        # Close progress window after completion
        progress_window.destroy()

        # Display the classified image
        self.display_image()
        tk.messagebox.showinfo("Testing Complete", "Testing process finished successfully!")

    def save_nn(self):
        """Save the trained model to disk."""
        try:
            if self.model is None:
                tk.messagebox.showerror("Error", "No model is initialized. Please build a model first.")
                return

            # Save the model
            file_path = filedialog.asksaveasfilename(
                defaultextension=".h5",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
            )
            if file_path:
                self.model.save(file_path)
                tk.messagebox.showinfo("Model Saved", f"Model has been saved to:\n{file_path}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred during saving:\n{str(e)}")

    def start_train_cnn(self):
        """Start training the CNN model."""

        if not hasattr(self, "model"):
            print("CNN model is not created yet!")
            return

        # Extract training parameters
        batch_size = int(self.batch_size_value)
        epochs = int(self.epoch_value)

        # Callbacks (optional): Stop training early if no improvement
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        # model_checkpoint = ModelCheckpoint("best_cnn_model.h5", save_best_only=True, monitor="val_loss")

        # Create a training progress callback that writes to summary_text
        progress_callback = TrainingProgressCallback(self.summary_text)

        # Train CNN
        print(f"Starting CNN Training: Batch Size={batch_size}, Epochs={epochs}")

        history = self.model.fit(
            self.train_x, self.train_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,  # Use 20% of data for validation
            # callbacks=[early_stopping, model_checkpoint]
            callbacks=[early_stopping, progress_callback]
        )

        print("Training complete!")

    def test_cnn(self):
        """Test the trained CNN model on the entire spectral image with a progress bar."""
        if self.model is None:
            tk.messagebox.showerror("Error", "No CNN model is trained yet. Train the model first.")
            return

        if self.spec_image is None:
            tk.messagebox.showerror("Error", "No spectral image loaded!")
            return

        spectral_data = np.array(self.spec_image)  # Shape (height, width, num_bands)
        print(f"spectral_data.shape: {spectral_data.shape}")

        height, width, num_bands = spectral_data.shape

        # Normalize the spectral data
        range_value = self.max_value - self.min_value
        spectral_data = (spectral_data - self.min_value) / (range_value + 1e-10)  # Avoid division by zero

        # Compute padding size
        pad_size = self.patch_size // 2

        # Pad the spectral image for patch extraction
        padded_spec_image = np.pad(spectral_data,
                                   ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                                   mode='reflect')

        cnn_input_data = []
        total_patches = height * width  # Total number of patches

        # Create a new window for progress tracking
        progress_window = tk.Toplevel(self.summary_label)
        progress_window.title("Testing Progress")

        # Progress bar UI
        progress_label = tk.Label(progress_window, text="Processing image patches...", font=("Arial", 10))
        progress_label.pack(pady=5)

        progress_bar = ttk.Progressbar(progress_window, length=400, mode='determinate')
        progress_bar.pack(pady=10)
        progress_bar["maximum"] = total_patches  # Maximum progress value

        # Extract 5×5 patches
        patch_count = 0
        for y in range(height):
            for x in range(width):
                patch = padded_spec_image[y:y + self.patch_size, x:x + self.patch_size, :]
                if patch.shape == (self.patch_size, self.patch_size, num_bands):
                    cnn_input_data.append(patch)

                # Update progress bar
                patch_count += 1
                progress_bar["value"] = patch_count
                progress_window.update_idletasks()  # Keep UI responsive

        cnn_input_data = np.array(cnn_input_data)  # Convert list to array
        print(f"cnn_input_data.shape: {cnn_input_data.shape}")  # Should be (num_patches, 5, 5, num_bands)

        # Predict the class for each patch
        predictions = self.model.predict(cnn_input_data, batch_size=32)  # Use batch processing
        print(f"Testing Results Shape: {predictions.shape}")

        # Get color mappings
        colors = np.array([self.hex_to_rgb(color) for (_, _, _, _, color, _) in self.rectangles])
        colors = colors[:self.output_neurons]  # Adjust based on the number of classes

        # Reconstruct the classified image
        # self.rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.accuracy = float(self.test_accuracy_value.get())

        patch_index = 0
        for y in range(height):
            for x in range(width):
                max_value_index = np.argmax(predictions[patch_index])  # Get predicted class
                max_value = predictions[patch_index][max_value_index]

                # Assign color if confidence is > 0.1
                if max_value > self.accuracy :
                    self.rgb_image[y, x] = colors[max_value_index]

                patch_index += 1

                # Update progress bar
                progress_bar["value"] = patch_index
                progress_window.update_idletasks()  # Refresh UI

        # Close progress window when done
        progress_window.destroy()

        # Display the classified image
        self.display_image()
        tk.messagebox.showinfo("Testing Complete", "CNN Testing process finished successfully!")

    def save_cnn(self):
        """Save the trained CNN model to a file."""

        if not hasattr(self, "model"):
            print("No trained model to save!")
            return

        # Ask the user where to save the model
        file_path = filedialog.asksaveasfilename(defaultextension=".h5",
                                                 filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")])
        if file_path:
            self.model.save(file_path)
            print(f"Model saved at: {file_path}")

class StopTrainingCallback(tensorflow.keras.callbacks.Callback):
    """Callback to stop training."""
    def __init__(self):
        super().__init__()
        self.stop = False

    def on_batch_end(self, batch, logs=None):
        if self.stop:
            self.model.stop_training = True

class TrainingProgressCallback(Callback):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        log_message = f"Epoch {epoch + 1}: Loss={logs.get('loss', 0):.4f}, Accuracy={logs.get('accuracy', 0):.4f}, Val_Loss={logs.get('val_loss', 0):.4f}, Val_Accuracy={logs.get('val_accuracy', 0):.4f}\n"

        # Append text to summary_text widget
        self.text_widget.insert(tk.END, log_message)
        self.text_widget.see(tk.END)  # Auto-scroll to the bottom

        # Force Tkinter update
        self.text_widget.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('1400x1000')
    app = App(root)
    root.mainloop()
