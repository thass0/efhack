import pygame
import numpy as np
import pygame_gui
import renderer
import sim_point
from PIL import Image
import random

width, height = 175, 175
upsampling = 3

# Open the image
images_path = ["neptunemap.jpg", "mars_1k_color.jpg", "mercurymap.jpg", "plutomap1k.jpg"]
# image_path = images_path[random.randint(0, len(images_path)-1)]  # Replace with your image path
image_path = "neptunemap.jpg"
image = Image.open(image_path)

# Convert the image to RGB if it's not already
image = image.convert("RGB")
image = image.resize((width,height),Image.Resampling.NEAREST)

# Get the image size (width and height)
img_width, img_height = image.size

# Access pixels using the load() method
img_pixels = np.array(image)
img_pixels = img_pixels/25

# Initialize Pygame
pygame.init()

# Set up the display window

screen = pygame.display.set_mode((width*upsampling, height*upsampling))
pygame.display.set_caption("Fast Pixel Rendering")

# Create a manager for the UI elements
manager = pygame_gui.UIManager((width, height))

# Create a slider (to control brightness)
brightness_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(10, 10, 200, 20),
    start_value=1.0,  # Start at 1 (full brightness)
    value_range=(0.0, 1.0),
    manager=manager
)

# Define a clock to manage the frame rate
clock = pygame.time.Clock()

# Create a NumPy array to hold pixel values
pixels = np.zeros((width, height, 3), dtype=np.uint8)

cam_pos = np.array([0,0,-300])
cam_dis = 50
plan_pos = np.array([0,0,0])
distance_between_pixels = 1
displacement_width = (1/2)*width*distance_between_pixels
displacement_height = (1/2)*height*distance_between_pixels
distance_between_points = 1

def ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius):
    # Vector from the ray's origin to the sphere's center
    oc = ray_origin - sphere_center

    # Calculate coefficients for the quadratic equation At^2 + Bt + C = 0
    a = np.dot(ray_direction, ray_direction)  # Should be 1 if the direction is normalized
    b = 2 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere_radius ** 2

    # Calculate the discriminant
    discriminant = b ** 2 - 4 * a * c

    # If the discriminant is negative, there are no real atm_interesctions
    if discriminant < 0:
        return None

    # Calculate the two possible solutions (atm_interesctions)
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)

    # Return the intersection points (if t1 and t2 are positive, they are along the ray direction)
    atm_interesctions = []
    if t1 >= 0:
        atm_interesctions.append(ray_origin + t1 * ray_direction)
    if t2 >= 0:
        atm_interesctions.append(ray_origin + t2 * ray_direction)

    return atm_interesctions if atm_interesctions else None

def add_stars(pixels, num_stars=100):
    """ Add random stars to the background """
    for _ in range(num_stars):
        star_x = np.random.randint(0, width)
        star_y = np.random.randint(0, height)
        star_brightness = np.random.randint(200, 255)  # Brightness of stars
        pixels[star_x, star_y] = (255, 255, 225)  # White stars
    return pixels

# Main loop to handle rendering and updates
running = True
while running:
    time_delta = clock.tick(60) / 1000.0  # Time in seconds
    # Handle events (e.g., closing the window)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        manager.process_events(event)  # Process events for the UI elements

    add_stars(pixels)
    # Get the current value of the brightness from the slider
    brightness = brightness_slider.get_current_value()

    # Define the intensity and wavelength values
    intensity = np.array([1.10, 1.25, 1.30, 1.28, 1, 1.15, 1, 1, 0.90, 0.85])
    intensity = intensity / 200 * brightness
    wavelength = np.array([400, 450, 500, 550, 600, 650, 700, 750, 800, 850])

    # Convert the intensity and wavelength to RGB values
    rgb_value = renderer.intensity_to_rgb(intensity, wavelength)

    

    # Example: Update a few random pixels with different colors
    for x in range(width):
        for y in range(height):
            grid_x_pos = -displacement_width + x*distance_between_pixels
            grid_y_pos = -displacement_height + y*distance_between_pixels
            grid_pos = np.array([grid_x_pos, grid_y_pos, cam_dis])
            dir_vec = grid_pos - cam_pos
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            atm_interesctions = ray_sphere_intersection(cam_pos, dir_vec, plan_pos, 70)
            plan_interesctions = ray_sphere_intersection(cam_pos, dir_vec, plan_pos, 50)
            

            if atm_interesctions != None and len(atm_interesctions) >= 2:
                if(plan_interesctions != None):
                    # print("found interesction")
                    distance = np.linalg.norm(plan_interesctions[0]-atm_interesctions[0])
                    amount_points = int(distance / distance_between_points)
                    heights = np.array([])
                    for i in range(amount_points):
                        heights = np.append(heights, np.linalg.norm((atm_interesctions[0] + i*dir_vec*distance_between_points) - plan_pos))
                    offset = (np.abs(width/2 - x)/width/2)**2  + (np.abs(height/2 - y)/height/2)**2
                    offset *= 100
                    # 
                    spectrum = sim_point.sim_point(2500.0, 1.0,[(sim_point.Element.HELIUM, 0.2), (sim_point.Element.OXYGEN, 0.0), (sim_point.Element.IRON, 0.8)], wavelength, np.array([1.10, img_pixels[x,y,2] - offset, 1.30, img_pixels[x,y,1] - offset, img_pixels[x,y,0] - offset, 1.15, 1, 1, 0.90, 0.85])*5.0e-37, heights, distance_between_points)
                    spectrum = np.array(spectrum)*5.0e32
                    #print(spectrum)
                    color = renderer.intensity_to_rgb(spectrum, wavelength)
                    #print(color)
                    pixels[x, y] = color

                else:
                    # print("found interesction")
                    distance = np.linalg.norm(atm_interesctions[1]-atm_interesctions[0])
                    amount_points = int(distance / distance_between_points)
                    heights = np.array([])
                    for i in range(amount_points):
                        heights = np.append(heights, np.linalg.norm((atm_interesctions[0] + i*dir_vec*distance_between_points) - plan_pos))
                    spectrum = sim_point.sim_point(2500.0, 1.0, [(sim_point.Element.HELIUM, 0.1), (sim_point.Element.OXYGEN, 0.1), (sim_point.Element.IRON, 0.8)], wavelength, [1.0e-36 for _ in range(len(wavelength))], heights, distance_between_points)
                    spectrum = np.array(spectrum)*5.0e32
                    # print(spectrum)
                    color = renderer.intensity_to_rgb(spectrum, wavelength)
                    #print(color)
                    pixels[x, y] = color

    # print(pixels)
    # Convert the NumPy array to a surface for Pygame
    pixels_upsampled = np.repeat(pixels, upsampling, axis=0)  # Duplicate along the rows
    pixels_upsampled = np.repeat(pixels_upsampled, upsampling, axis=1)  # Duplicate along the columns
    pixel_surface = pygame.surfarray.make_surface(pixels_upsampled)

    # Render the surface onto the screen
    screen.blit(pixel_surface, (0, 0))

    # Update the UI elements (like the slider)
    # manager.update(time_delta)

    # Draw the UI elements (including the slider)
    # manager.draw_ui(screen)

    # Update the screen
    pygame.display.flip()
    clock.tick(1)  # Force 30 FPS
    
    break

print("done")
while True:
    continue
# Quit Pygame
pygame.quit()
