import pygame
import numpy as np
import pygame_gui
import renderer


# Initialize Pygame
pygame.init()

# Set up the display window
width, height = 200, 200
screen = pygame.display.set_mode((width, height))
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
cam_dis = 100
plan_pos = np.array([0,0,0])
distance_between_pixels = 1
displacement_width = (1/2)*width*distance_between_pixels
displacement_height = (1/2)*height*distance_between_pixels
distance_between_points = 10

def ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius):
    # Vector from the ray's origin to the sphere's center
    oc = ray_origin - sphere_center

    # Calculate coefficients for the quadratic equation At^2 + Bt + C = 0
    a = np.dot(ray_direction, ray_direction)  # Should be 1 if the direction is normalized
    b = 2 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere_radius ** 2

    # Calculate the discriminant
    discriminant = b ** 2 - 4 * a * c

    # If the discriminant is negative, there are no real intersections
    if discriminant < 0:
        return None

    # Calculate the two possible solutions (intersections)
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)

    # Return the intersection points (if t1 and t2 are positive, they are along the ray direction)
    intersections = []
    if t1 >= 0:
        intersections.append(ray_origin + t1 * ray_direction)
    if t2 >= 0:
        intersections.append(ray_origin + t2 * ray_direction)

    return intersections if intersections else None

# Main loop to handle rendering and updates
running = True
while running:
    time_delta = clock.tick(60) / 1000.0  # Time in seconds
    # Handle events (e.g., closing the window)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        manager.process_events(event)  # Process events for the UI elements

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
            intersections = ray_sphere_intersection(cam_pos, dir_vec, plan_pos, 100)
            if len(intersections) >= 2:
                distance = np.linalg.norm(intersections[1]-intersections[0])
                amount_points = int(distance / distance_between_points)
                heights = np.array([])
                for i in range(amount_points):
                    heights = np.append(heights, np.linalg.norm((intersections[0] + i*dir_vec*distance_between_points) - plan_pos))
            pixels[x, y] = (0,0,0)


    # Convert the NumPy array to a surface for Pygame
    pixel_surface = pygame.surfarray.make_surface(pixels)

    # Render the surface onto the screen
    screen.blit(pixel_surface, (0, 0))

    # Update the UI elements (like the slider)
    manager.update(time_delta)

    # Draw the UI elements (including the slider)
    manager.draw_ui(screen)

    # Update the screen
    pygame.display.flip()
    clock.tick(1)  # Force 30 FPS

# Quit Pygame
pygame.quit()
