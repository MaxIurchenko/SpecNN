import tkinter as tk
from tkinter import ttk


class NeuralNetworkConfigTable:
    def __init__(self, root, train_x_shape, train_y_shape):
        """Initialize the table with train_x and train_y data shapes."""
        self.root = root
        self.root.title("Neural Network Configuration")

        # Store the training data shapes
        self.train_x_shape = train_x_shape
        self.train_y_shape = train_y_shape

        # Create table headers
        headers = ["Layer", "Configuration"]
        for col, header in enumerate(headers):
            label = tk.Label(root, text=header, font=("Arial", 12, "bold"), bg="lightgray")
            label.grid(row=0, column=col, sticky="nsew", padx=2, pady=2)

        # Create the table rows
        self.create_table_row(1, "Input shape", f"{train_x_shape}", readonly=True)
        self.first_hidden_layer = self.create_table_row(2, "First Hidden Layer", "", input_type="int")
        self.second_hidden_layer = self.create_table_row(3, "Second Hidden Layer", "", input_type="int")
        self.output_layer = self.create_table_row(4, "Output Layer", f"{train_y_shape[1]}", readonly=True)

        # Add a button to print the configuration
        submit_btn = tk.Button(root, text="Submit", command=self.print_configuration, font=("Arial", 12))
        submit_btn.grid(row=5, column=0, columnspan=2, pady=10)

    def create_table_row(self, row, label_text, default_value, input_type=None, readonly=False):
        """Helper function to create a row in the table."""
        label = tk.Label(self.root, text=label_text, font=("Arial", 12))
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)

        if readonly:
            # Create a label for readonly fields
            entry = tk.Label(self.root, text=default_value, font=("Arial", 12), bg="white", relief="solid")
            entry.grid(row=row, column=1, sticky="nsew", padx=5, pady=5)
        else:
            # Create an entry for user input
            entry = ttk.Entry(self.root, font=("Arial", 12))
            entry.grid(row=row, column=1, sticky="nsew", padx=5, pady=5)
            if input_type == "int":
                entry.insert(0, "0")  # Default integer value

        return entry

    def print_configuration(self):
        """Print the neural network configuration."""
        try:
            # Retrieve user inputs
            first_hidden = int(self.first_hidden_layer.get())
            second_hidden = int(self.second_hidden_layer.get())

            print("Neural Network Configuration:")
            print(f"Input Shape: {self.train_x_shape}")
            print(f"First Hidden Layer: {first_hidden} neurons")
            print(f"Second Hidden Layer: {second_hidden} neurons")
            print(f"Output Layer: {self.train_y_shape[1]} classes")
        except ValueError:
            print("Please enter valid integers for the hidden layers.")


# Example usage
if __name__ == "__main__":
    # Example shapes for train_x and train_y
    train_x_shape = (572, 135)
    train_y_shape = (572, 10)

    root = tk.Tk()
    app = NeuralNetworkConfigTable(root, train_x_shape, train_y_shape)
    root.mainloop()
