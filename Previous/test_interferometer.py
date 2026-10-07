import numpy as np
import matplotlib.pyplot as plt

# Parameters
wavelength = 632.8e-9  # HeNe laser wavelength in meters (632.8 nm)
k = 2 * np.pi / wavelength  # Wave number
grid_size = 500  # Number of pixels in the image
extent = 0.001  # Physical size of the detector in meters (10 mm)

# Create a coordinate grid (x, y) on the camera sensor
x = np.linspace(-extent/2, extent/2, grid_size)
y = np.linspace(-extent/2, extent/2, grid_size)
X, Y = np.meshgrid(x, y)

# Simulate a slight tilt (angle alpha) between the two wavefronts
# This creates spatial parallel interference fringes
alpha = 0.0005  # Tilt angle in radians
optical_path_diff = 2 * X * np.sin(alpha)

# Calculate phase difference and intensity (cosine squared interference law)
phase_diff = k * optical_path_diff
intensity = np.cos(phase_diff / 2)**2

# Plot the fringe pattern
plt.figure(figsize=(6, 5))
plt.imshow(intensity, extent=[-extent*1000/2, extent*1000/2, -extent*1000/2, extent*1000/2], cmap='gray')
plt.title('Simulated Interferometer Fringes')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.colorbar(label='Intensity (Normalized)')
plt.show()
