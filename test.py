from PIL import Image
import numpy as np

# Open the image
image_path = "mars_1k_color.jpg"  # Replace with your image path
image = Image.open(image_path)

# Convert the image to RGB if it's not already
image = image.convert("RGB")

# Get the image size (width and height)
img_width, img_height = image.size

# Access pixels using the load() method
img_pixels = np.array(image)

# Example: Print the RGB value of the pixel at (x, y)
x, y = 50, 50  # Example coordinates (change as needed)
rgb_value = pixels[x, y]
print(f"RGB value at ({x}, {y}): {rgb_value}")

# Example: Convert the image to a NumPy array for easier manipulation
image_array = np.array(image)
print(f"Image as NumPy array: {image_array.shape}")

# Access a pixel using NumPy array indexing
rgb_value_np = image_array[50, 50]  # Same coordinates as above
print(f"RGB value from NumPy array at (50, 50): {rgb_value_np}")
