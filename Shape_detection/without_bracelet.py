import numpy as np
import time
import sys
                                    
# Define shapes with vertices
shapes = {
    'arrow': [(0, 0), (3, 0), (3, 1), (5, -1), (3, -3), (3, -2), (0, -2)],
    'cross': [(0, 0), (2, 0), (2, 2), (4, 2), (4, 0), (6, 0), (6, -2), (4, -2), (4, -4), (2, -4), (2, -2), (0, -2)],
    'hexagon': [(0, 0), (3, 2), (6, 0), (6, -3), (3, -5), (0, -3)],
    'kite': [(0, 0), (2, 2), (4, 0), (2, -5)],
    'octagon': [(0, 0), (2, 2), (4, 2), (6, 0), (6, -2), (4, -4), (2, -4), (0,-2)],
    'parallelogram': [(0, 0), (2, 2), (6, 2), (4, 0)],
    'pentagon': [(0, 0), (2, 2), (4, 0), (3, -2), (1, -2)],
    'rhombus': [(0, 0), (2, 2), (4, 2), (2, 0)],
    'star': [(0, 0), (2, 0), (3, 2), (4, 0), (6, 0), (4, -1), (5, -3), (3, -2), (1, -3), (2, -1)],
    'trapezoid': [(0, 0), (1, 2), (4, 2), (5, 0)],
    'square': [(0, 0), (0, 2), (2, 2), (2, 0)],
    'rectangle': [(0, 0), (0, 2), (4, 2), (4, 0)],
    'triangle': [(0, 0), (3, 3), (3, 0)],
    'diamond' :[(0,0), (1,2), (3,2), (4,0), (2,-4)],
    'one' : [(0,0), (0,-4)],
    'two' : [(0,0), (2,0), (2,-2), (0,-2), (0,-4), (2,-4)],
    'three' : [(0,0), (2,0), (2,-2), (0,-2), (2,-2), (2,-4), (0,-4)],
    'four' : [(0,0), (0,-2), (2,-2), (2,0), (2,-4)],
    'five' : [(0,0), (-2,0), (-2,-2), (0,-2), (0,-4), (-2,-4)],
    'six' : [(0,0), (-2,0), (-2,-4), (0,-4), (0,-2), (-2,-2)],
    'seven' : [(0,0), (2,0), (2,-4)],
    'eight' : [(0,0), (-2,0), (-2,-2), (0,-2), (0,-4), (-2,-4), (-2,-2), (0,-2), (0,0)],
    'nine' : [(0,0), (-2,0), (-2,-2), (0,-2), (0,0), (0,-4), (-2,-4)],
    'c' : [(0,0), (-2,0), (-2,-4), (0,-4)],
    'e' : [(0,0), (-2,0), (-2,-2), (0,-2), (-2,-2), (-2,-4), (0,-4)],
    'j' : [(0,0), (2,0), (2,-4), (0,-4), (0,-2)],
    'l' : [(0,0), (0,-4), (2,-4)],
    'm' : [(0,0), (0,4), (1,4), (1,0), (1,4), (2,4), (2,0)],
    'n' : [(0,0), (0,4), (2,4), (2,0)],
    'p' : [(0,0), (0,4), (2,4), (2,2), (0,2)],
    'u' : [(0,0), (0,-4), (2,-4), (2,0)],
    'r' : [(0,0), (0,4), (2,4), (2,2), (0,2), (2,0)],
    'v' : [(0,0), (2,-4), (4,0)],
    'w' : [(0,0), (0,-4), (2,-2), (4,-4), (4,0)],
    'z' : [(0,0), (2,0), (0,-4), (2,-4)],
}

# Function to calculate direction and distance
def calculate_direction_and_time(start, end, speed=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = np.sqrt(dx**2 + dy**2)
    time_required = distance / speed 

    intensity = 50
    
    if dx > 0 and dy == 0:
            return 'right', time_required
    elif dx < 0 and dy == 0:
            return 'left', time_required
    elif dy > 0 and dx == 0:  
            return 'top', time_required
    elif dy < 0 and dx == 0:
            return 'down', time_required
    elif dx > 0 and dy > 0:
            return 'diagonal right top', time_required
    elif dx > 0 and dy < 0:
            return 'diagonal right bottom', time_required
    elif dx < 0 and dy > 0:
            return 'diagonal left top', time_required
    elif dx < 0 and dy < 0:
            return 'diagonal left bottom', time_required
    else:
        return 'none', 0
    
# Function to detect outline and simulate tactile feedback
def simulate_tactile_feedback(shape, speed=1):
    vertices = shapes[shape]
    if shape in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 
                 'c', 'e', 'j', 'l', 'm', 'n', 'p', 'u', 'r', 'v', 'w', 'z']:
        vertices.append(vertices[-1])  # Add the last vertex again to complete the shape
    else:
        vertices.append(vertices[0])  # Close the shape by returning to the starting point
    
    for i in range(len(vertices) - 1):
        start = vertices[i]
        end = vertices[i + 1]
        direction, time_required = calculate_direction_and_time(start, end, speed)
        if direction != 'none':
            print(f"Move {direction} for {time_required:.2f} seconds")
            time.sleep(time_required)  # Simulate the time required for the movement

# List of shapes to loop through
shapes_to_detect_1 = ['square', 'octagon', 'cross', 'seven', 'diamond', 'one', 'triangle', 
                      'two', 'w', 'kite', 'rectangle', 'nine', 'c', 'j', 'four', 'star', 'n', 
                      'pentagon', 'p', 'z', 'hexagon', 'u', 'l', 'three', 'v', 'six', 'e', 
                      'rhombus', 'm', 'eight', 'five', 'parallelogram', 'r', 'arrow', 'trapezoid']

shapes_to_detect_2 = ['parallelogram', 'e', 'arrow', 'r', 'six', 'p', 'one', 'seven', 'square', 
                      'u', 'n', 'pentagon', 'diamond', 'j', 'three', 'v', 'triangle', 'star', 'm', 
                      'five', 'rectangle', 'four', 'hexagon', 'kite', 'nine', 'octagon', 'eight', 
                      'w', 'trapezoid', 'cross', 'z', 'rhombus', 'c', 'l', 'two']

shapes_to_detect_3 = ['two', 'hexagon', 'n', 'l', 'cross', 'arrow', 'r', 'nine', 'eight', 'm', 
                      'seven', 'kite', 'rectangle', 'c', 'three', 'u', 'rhombus', 'five', 'star', 
                      'six', 'e', 'diamond', 'square', 'j', 'parallelogram', 'trapezoid', 'pentagon', 
                      'w', 'octagon', 'p', 'z', 'four', 'one', 'v', 'triangle']

# Loop through the shapes
for index, shape in enumerate(shapes_to_detect_1):
    print(shape)
    simulate_tactile_feedback(shape)
    print("stop \n")  # Adding a newline for better readability between shapes
    time.sleep(3)  # Pause for 3 seconds after each shape
    
    # Add a 5-second rest after every 5 shapes
    if (index + 1) % 5 == 0:
        print("5-second rest \n")
        time.sleep(5)


# Example usage
# shape_to_detect = 'nine'  # Change to 'rectangle', 'triangle', 'polygon' as needed
# simulate_tactile_feedback(shape_to_detect)