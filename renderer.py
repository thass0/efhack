import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt

filename = "cie-cmf.txt"

# Function to read CIE CMF data from the file
def read_cie_data(filename):
    wavelengths = []
    x_bar = []
    y_bar = []
    z_bar = []
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            wavelengths.append(float(parts[0]))
            x_bar.append(float(parts[1]))
            y_bar.append(float(parts[2]))
            z_bar.append(float(parts[3]))
    
    return np.array(wavelengths), np.array(x_bar), np.array(y_bar), np.array(z_bar)

# Read the CIE CMF data
wavelengths_cie, x_bar, y_bar, z_bar = read_cie_data("cie-cmf.txt")


def intensity_to_rgb(intensity, wavelength_in):
    x_interpolated = np.interp(wavelength_in, wavelengths_cie, x_bar)
    y_interpolated = np.interp(wavelength_in, wavelengths_cie, y_bar)
    z_interpolated = np.interp(wavelength_in, wavelengths_cie, z_bar)

    # Multiply intensity by the CIE functions and integrate
    X = integrate.simpson(x_interpolated * intensity, wavelength_in)
    Y = integrate.simpson(y_interpolated * intensity, wavelength_in)
    Z = integrate.simpson(z_interpolated * intensity, wavelength_in)

    # Convert XYZ to RGB (using sRGB conversion matrix)
    xyz_to_rgb_matrix = np.array([[3.2406, -1.5372, -0.4986],
                                [-0.9689, 1.8758, 0.0415],
                                [0.0557, -0.2040, 1.0570]])

    rgb = np.dot(xyz_to_rgb_matrix, np.array([X, Y, Z]))

    # Clip values to be between 0 and 1
    rgb = np.clip(rgb, 0, 1)

    # Convert to 8-bit RGB (0-255 range)
    return (rgb * 255).astype(int)


# print(f"RGB Value: {rgb_value}")
# Create a 1x1 image (a single pixel)
# image = np.array([[rgb_value]])

# Display the color using imshow
# plt.imshow(image)
# plt.axis('off')  # Hide the axis for a cleaner view
# plt.show()