import tkinter as tk
from tkinter import filedialog, colorchooser
from PIL import Image, ImageTk


class ImageSquareSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Square Selector")

        # Canvas for displaying image
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack(fill="both", expand=True)

        # Buttons for loading image and saving selections
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(side="left", padx=5, pady=5)

        self.save_button = tk.Button(root, text="Save Selections", command=self.save_selections)
        self.save_button.pack(side="left", padx=5, pady=5)

        # Variables for drawing
        self.image = None
        self.tk_image = None
        self.start_x = None
        self.start_y = None
        self.rectangles = []  # Stores rectangles as (x1, y1, x2, y2, color)
        self.current_rectangle = None

        # Bind mouse events to canvas
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.complete_rectangle)

    def load_image(self):
        """Load an image and display it on the canvas."""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image Files", "*.png;*.jpg;*.jpeg;*.tiff;*.bmp"),  # Specify common image formats
                ("All Files", "*.*"),
            ]
        )
        if file_path:
            self.image = Image.open(file_path)
            self.image = self.image.resize((800, 600), Image.ANTIALIAS)  # Resize to fit canvas
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            print(f"Loaded image: {file_path}")

    def start_draw(self, event):
        """Start drawing a rectangle."""
        self.start_x = event.x
        self.start_y = event.y
        self.current_rectangle = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y,
                                                              outline="red", width=2)

    def draw_rectangle(self, event):
        """Update the rectangle as the user drags the mouse."""
        if self.current_rectangle:
            self.canvas.coords(self.current_rectangle, self.start_x, self.start_y, event.x, event.y)

    def complete_rectangle(self, event):
        """Complete the rectangle and allow the user to choose a color."""
        if self.current_rectangle:
            x1, y1, x2, y2 = self.canvas.coords(self.current_rectangle)
            color = self.choose_color()
            if color:
                self.canvas.itemconfig(self.current_rectangle, outline=color)
                self.rectangles.append((x1, y1, x2, y2, color))
            else:
                self.canvas.delete(self.current_rectangle)
            self.current_rectangle = None

    def choose_color(self):
        """Allow the user to choose a color for the rectangle."""
        color = tk.colorchooser.askcolor(title="Choose Rectangle Color")[1]
        return color

    def save_selections(self):
        """Save the selected rectangles and their colors."""
        if self.rectangles:
            print("Selected Rectangles:")
            for rect in self.rectangles:
                print(f"Coordinates: {rect[:4]}, Color: {rect[4]}")
            print("Selections saved successfully.")
        else:
            print("No rectangles selected.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSquareSelector(root)
    root.mainloop()
