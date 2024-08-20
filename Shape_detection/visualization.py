import matplotlib.pyplot as plt
import numpy as np
import os

# Define shapes with vertices
shapes = {
    '0': [(0, 0), (2, 0), (2, -4), (0, -4),  (0, 0)],
    '1': [(0, 0), (0, -2)],
    '2': [(0, 0), (2, 0), (2, -2), (0, -2), (0, -4), (2, -4)],
    '3': [(0, 0), (2, 0), (2, -2), (0, -2), (2, -2), (2, -4), (0, -4)],
    '4': [(0, 0), (0, -2), (2, -2), (2, 0), (2, -4)],
    '5': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, -4), (-2, -4)],
    '6': [(0, 0), (-2, 0), (-2, -4), (0, -4), (0, -2), (-2, -2)],
    '7': [(0, 0), (2, 0), (2, -4)],
    '8': [(0, 0), (2, 0), (2, -2), (0, -2), (0, -4), (2, -4), (2, -2), (0, -2), (0, 0)],
    '9': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, 0), (0, -4), (-2, -4)],
    'a': [(0, 0), (-2, 0), (-2, 2), (0, 2), (0, -0.2)],
    'b': [(0, 0), (2, 0), (2, -2), (0, -2), (0, 2)],
    'c': [(0, 0), (-2, 0), (-2, -2), (0, -2)],
    'd': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, 2)],
    'e': [(0, 0), (2, 0), (2, 2), (0, 2), (0, -2), (2, -2)],
    'f': [(0, 0), (-2, 0), (-2, -4), (-2, -2), (0, -2)],
    'h': [(0, 0), (0, -4), (0, -2), (2, -2), (2, -4)],
    'i': [(0, 0), (2, 0), (1, 0), (1, -4), (0, -4), (2, -4)],
    'j': [(0, 0), (2, 0), (2, -4), (0, -4), (0, -2)],
    'k': [(0, 0), (0, -4), (2, -2), (1, -3), (2, -4)],
    'l': [(0, 0), (0, -4), (2, -4)],
    'm': [(0, 0), (0, 4), (2, 2), (4, 4), (4, 0)],
    'n': [(0, 0), (0, 4), (2, 0), (2, 4)],
    'p': [(0, 0), (2, 0), (2, 2), (0, 2), (0, -2)],
    'q': [(0, 0), (-2, 0), (-2, 2), (0, 2), (0, -2)],
    'u': [(0, 0), (0, -2), (2, -2), (2, 0)],
    'r': [(0, 0), (2, 0), (2, 2), (0, 2), (0, -2), (0,0), (2,-2)],
    'v': [(0, 0), (2, -4), (4, 0)],
    'w': [(0, 0), (0, -4), (2, -2), (4, -4), (4, 0)],
    'y': [(0, 0), (2, -2), (4, 0), (2, -2), (2, -4)],
    'z': [(0, 0), (2, 0), (0, -2), (2, -2)]
}

# Function to visualize and save shapes with arrows
def visualize_and_save_shapes_with_arrows(shapes, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for shape_name, vertices in shapes.items():
        plt.figure()
        plt.axis('off')  # Turn off the axis
        
        for i in range(len(vertices) - 1):
            start = vertices[i]
            end = vertices[i + 1]
            plt.plot([start[0], end[0]], [start[1], end[1]], marker='o')
            # with arrow
            #plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
            #          head_width=0.2, length_includes_head=True, color='blue')

        plt.gca().set_aspect('equal', adjustable='box')
        
        # Save the figure as a JPEG file
        plt.savefig(os.path.join(save_path, f"{shape_name}.jpg"), bbox_inches='tight', pad_inches=0, format='jpg')
        plt.close()

# Path to save the figures
save_path = r"D:\WWU\M8 - Master Thesis\Project\Code\Images"

# Visualize and save all shapes with arrows
visualize_and_save_shapes_with_arrows(shapes, save_path)
