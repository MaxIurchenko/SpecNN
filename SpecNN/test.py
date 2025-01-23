import tensorflow as tf
import numpy as np

# Example data
n_samples = 1000  # Number of spectra
spectrum_length = 50  # Length of each spectrum

# Create train_x (1000 spectra, each with 50 values between 0 and 1)
train_x = np.random.rand(n_samples, spectrum_length).astype(np.float32)
print(train_x[1])

# Create train_y (1000 labels, each a value between 0 and 50)
train_y = np.random.randint(0, 51, size=(n_samples,), dtype=np.int32)
print(train_y)

# Create the dataset
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))

# Inspect the dataset
for x, y in dataset.take(1):
    print("Spectrum (train_x):", x)
    print("Label (train_y):", y)