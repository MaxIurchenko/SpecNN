import numpy as np
import matplotlib.pyplot as plt


# Load the .npz file
data = np.load("222.npz")

# List all arrays stored in the file
print(data.files)

# Access a specific array by name
array_name = data.files[0]  # Example: Get the first array name
array_data = data[array_name]

train_x = data["train_x"]
train_y = data["train_y"]



# Print or use the array
print(array_data.shape)
print(array_data)
print(train_x)
print(train_y)

num_samples, num_bands = train_x.shape

# Plot all spectra
plt.figure(figsize=(10, 6))

for i in range(num_samples):
    plt.plot(range(num_bands), train_x[i], alpha=0.5)  # alpha for transparency

# Labels and title
plt.xlabel("Spectral Band")
plt.ylabel("Reflectance / Intensity")
plt.title("Spectral Signatures of Training Samples")
plt.show()