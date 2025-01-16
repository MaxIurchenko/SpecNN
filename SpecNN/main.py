import tkinter as tk
from tkinter import filedialog, colorchooser, ttk
# from unittest.mock import right

import spectral
import numpy as np
from PIL import Image, ImageTk, ImageGrab
from tensorflow.python.keras.saving.saved_model.save_impl import input_layer
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt

# import pandas as pd
# import cv2



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

        # Top menu -----------------------------------------------------------------------------------------------
        self.top_menu = tk.Label(self.frame)
        self.top_menu.config(justify='left', background="lightgray", height=2)
        self.top_menu.grid(row=0, column=0, columnspan=2, sticky='nwe')

        self.open_file_button = tk.Button(self.top_menu, text="Open HDR", command=self.open_file)
        self.open_file_button.pack(side="left", padx=0, pady=0)

        # Bind the resize event and mouse motion
        self.toggle_button = tk.Button(self.top_menu, text="Toggle Magnifying Glass", command=self.toggle_magnifier)
        self.toggle_button.pack(pady=0, side='left')

        # Image display-----------------------------------------------------------------------------------------------
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

        # Bind this method to the treeview
        self.rect_table.bind("<Double-1>", self.edit_data_cell)

        # Buttons for table actions
        self.delete_button = tk.Button(self.right_label, text="Delete Selected", command=self.delete_rectangle)
        self.delete_button.grid(row=1, column=0, sticky="nwe")

        # Button save preparing data
        self.delete_button = tk.Button(self.right_label, text="Save train data", command=self.save_train_data)
        self.delete_button.grid(row=2, column=0, sticky="nwe")

        # Button save preparing data to file
        self.delete_button = tk.Button(self.right_label, text="Save train data to file", command=self.save_train_data_to_file)
        self.delete_button.grid(row=3, column=0, sticky="nwe")

        # Button load preparing data to file
        self.delete_button = tk.Button(self.right_label, text="Load train data", command=self.load_train_data)
        self.delete_button.grid(row=4, column=0, sticky="nwe")

        # Button clear preparing data to file
        self.delete_button = tk.Button(self.right_label, text="Clear train data", command=self.clear_train_data)
        self.delete_button.grid(row=5, column=0, sticky="nwe")

        # self.edit_button = tk.Button(self.right_label, text="Edit Selected", command=self.edit_rectangle)
        # self.edit_button.pack(fill=tk.X)

        # Label to display train_x and train_y shapes
        self.train_shape_label = tk.Label(self.right_label, text="train_x: N/A, train_y: N/A", font=("Arial", 12))
        self.train_shape_label.grid(row=6, column=0, sticky="nwe")

        #-------------------------------Neural Networks----------------------------------------
        self.nn_combobox = ttk.Combobox(self.right_label, values=["Simple MLP", "CNN", "RNN"], font=("Arial", 12), state='readonly')
        self.nn_combobox.grid(row=7, column=0, sticky="nwe")
        self.nn_combobox.bind("<<ComboboxSelected>>", self.nn_combobox_selected)

        self.nn_parameters = tk.Label(self.right_label)
        self.nn_parameters.grid(row=8, column=0, sticky="nwe")

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
            orig_x1 = int(x1 / self.scale_x)
            orig_y1 = int(y1 / self.scale_y)
            orig_x2 = int(x2 / self.scale_x)
            orig_y2 = int(y2 / self.scale_y)

            color = self.choose_color()
            if color:
                self.label_image.itemconfig(self.current_rectangle, outline=color)
                self.rectangles.append((orig_x1, orig_y1, orig_x2, orig_y2, color, 0))  # Store in original space
                self.rect_table.insert("", "end", values=(orig_x1, orig_y1, orig_x2, orig_y2, "", 0))
                iid = self.rect_table.get_children()  # First row
                if len(iid) > 0:
                    last_iid = iid[-1]
                else:
                    last_iid = iid[0]

                self.set_treeview_cell_color(self.rect_table, last_iid, colnum=4, bg_color=color)
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
            spectral_data = self.spec_image[y1:y2, x1:x2, :]  # Extract the region for all bands

            # Check the shape of the extracted spectral data (should be (height, width, num_bands))
            height, width, num_bands = spectral_data.shape

            # Flatten the spectral data into (height * width, num_bands)
            flattened_spectral_data = spectral_data.reshape(-1, num_bands)

            # Append the flattened data to training features
            self.train_x.append(flattened_spectral_data)

            # Append the label (which is stored in 'data')
            self.train_y.append(np.full(flattened_spectral_data.shape[0], data))  # One label for each pixel

        # Convert the training data to numpy arrays
        self.train_x = np.vstack(self.train_x)  # Stack the individual arrays vertically
        self.train_y = np.concatenate(self.train_y)  # Concatenate all labels

        self.update_shape_label()

        # Check the shape of the resulting arrays
        print(f"Training data generated: {self.train_x.shape}, {self.train_y.shape}")
        print(self.train_x[0], self.train_y)

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

    def clear_train_data(self):
        """Clear the training data."""
        if hasattr(self, 'train_x') or hasattr(self, 'train_y'):
            self.train_x = None
            self.train_y = None
            print("Training data has been cleared.")
        else:
            print("No training data to clear.")
        self.update_shape_label()

#----------------------NeuralNetworks------------------------------------
    def nn_combobox_selected(self, event):
        """Callback for when a neural network type is selected."""
        selection = self.nn_combobox.get()

        if selection == "Simple MLP":
            model = self.create_simple_mlp_parameters()
        elif selection == "CNN":
            model = self.create_cnn(input_shape=(32, 32, 3), num_classes=10)
        elif selection == "RNN":
            model = self.create_rnn(input_shape=(100, 1), num_classes=10)
        else:
            self.summary_label.insert("1.0", "Invalid selection.\n")
            return

    def create_simple_mlp_parameters(self):
        # Create table headers
        headers = ["Layer", "Configuration"]
        self.mlp_parametrs = tk.Label(self.right_label, font=("Arial", 12, "bold"))
        self.mlp_parametrs.grid(row=9, column=0, sticky="new")

        for col, header in enumerate(headers):
            label = tk.Label(self.mlp_parametrs, text=header, font=("Arial", 12, "bold"), bg="lightgray")
            label.grid(row=1, column=col, sticky="new", padx=2, pady=2)

        unique_values, self.counts = np.unique(self.train_y, return_counts=True)
        self.output_neurons = len(unique_values)

        # Create the table rows
        self.create_table_row(2, "Input layer", f"{self.train_x.shape[1]}", readonly=True)
        self.first_hidden_layer = self.create_table_row(3, "First Hidden Layer", "", input_type="int")
        self.second_hidden_layer = self.create_table_row(4, "Second Hidden Layer", "", input_type="int")
        self.output_layer = self.create_table_row(5, "Output Layer", f"{self.output_neurons}", readonly=True)
        self.epoch = self.create_table_row(6, "Epoch", "", input_type="int")
        self.batch_size = self.create_table_row(7, "Batch size", "", input_type="int")

        # Add a button to print the configuration
        submit_btn = tk.Button(self.mlp_parametrs, text="Submit", command=self.make_nn_model, font=("Arial", 12))
        submit_btn.grid(row=8, column=0, columnspan=2, pady=10)

    def create_table_row(self, row, label_text, default_value, input_type=None, readonly=False):
        """Helper function to create a row in the table."""
        label = tk.Label(self.mlp_parametrs, text=label_text, font=("Arial", 12))
        label.grid(row=row, column=0, sticky="w", padx=0, pady=0)

        if readonly:
            # Create a label for readonly fields
            entry = tk.Label(self.mlp_parametrs, text=default_value, font=("Arial", 12), bg="white", relief="solid")
            entry.grid(row=row, column=1, sticky="nsew", padx=0, pady=0)
        else:
            # Create an entry for user input
            entry = ttk.Entry(self.mlp_parametrs, font=("Arial", 12))
            entry.grid(row=row, column=1, sticky="nsew", padx=0, pady=0)
            if input_type == "int":
                entry.insert(0, "0")  # Default integer value

        return entry

    def make_nn_model(self):
        selection = self.nn_combobox.get()
        first_hidden = int(self.first_hidden_layer.get())
        second_hidden = int(self.second_hidden_layer.get())

        if selection == "Simple MLP":
            self.model = NN.create_simple_mlp(self.train_x.shape[1],
                                              int(self.first_hidden_layer.get()),
                                              int(self.second_hidden_layer.get()),
                                              self.output_neurons)

            print(self.model.summary())

        elif selection == "CNN":
            model = self.create_cnn(input_shape=(32, 32, 3), num_classes=10)
        elif selection == "RNN":
            model = self.create_rnn(input_shape=(100, 1), num_classes=10)
        else:
            self.summary_label.insert("1.0", "Invalid selection.\n")
            return


class NN:
    def create_simple_mlp(input_shape, first_layer, second_layer, output_layer):
        """Create a simple multi-layer perceptron (MLP) model."""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(first_layer, activation='relu'),
            layers.Dense(second_layer, activation='relu'),
            layers.Dense(output_layer, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    # def create_cnn(self, input_shape=(32, 32, 3), num_classes=10):
    #     """Create a convolutional neural network (CNN) model."""
    #     model = models.Sequential([
    #         layers.Input(shape=input_shape),
    #         layers.Conv2D(32, (3, 3), activation='relu'),
    #         layers.MaxPooling2D((2, 2)),
    #         layers.Conv2D(64, (3, 3), activation='relu'),
    #         layers.MaxPooling2D((2, 2)),
    #         layers.Flatten(),
    #         layers.Dense(64, activation='relu'),
    #         layers.Dense(num_classes, activation='softmax')
    #     ])
    #     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #     return model
    #
    # def create_rnn(self, input_shape=(100, 1), num_classes=10):
    #     """Create a recurrent neural network (RNN) model."""
    #     model = models.Sequential([
    #         layers.Input(shape=input_shape),
    #         layers.SimpleRNN(64, activation='relu', return_sequences=True),
    #         layers.SimpleRNN(64, activation='relu'),
    #         layers.Dense(num_classes, activation='softmax')
    #     ])
    #     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #     return model

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('1200x800')
    app = App(root)
    root.mainloop()
