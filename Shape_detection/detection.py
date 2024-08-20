import numpy as np
import time
import sys
import random
from auto_connect import interactive_belt_connect, setup_logger
from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption, BeltVibrationPattern)
from bracelet import connect_belt

# Connect to the belt
connection_check, belt_controller = connect_belt()
if connection_check:
    print('Bracelet connection successful.')
else:
    print('Error connecting bracelet. Aborting.')
    sys.exit()

# Define shapes with vertices
shapes = {
    '0': [(0, 0), (2, 0), (2, -4), (0, -4),  (0, 0)],
    '1': [(0, 0), (0, -2)],
    '2': [(0, 0), (2, 0), (2, -2), (0, -2), (0, -4), (2, -4)],
    '3': [(0, 0), (2, 0), (2, -2), (0, -2), (2, -2), (2, -4), (0, -4)],
    '4': [(0, 0), (0, -2), (2, -2), (2, 0), (2, -4)],
    '5': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, -4), (-2, -4)],
    '6': [(0, 0), (0, -4), (2, -4), (2, -2), (0, -2)],
    '7': [(0, 0), (2, 0), (2, -4)],
    '8': [(0, 0), (2, 0), (2, -2), (0, -2), (0, -4), (2, -4), (2, -2), (0, -2), (0, 0)],
    '9': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, 0), (0, -4)],
    'a': [(0, 0), (-2, 0), (-2, 2), (0, 2), (0, -2.2), (0.2, 2.2)],
    'b': [(0, 0), (2, 0), (2, -2), (0, -2), (0, 2)],
    'c': [(0, 0), (-2, 0), (-2, -2), (0, -2)],
    'd': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, 2)],
    'e': [(0, 0), (2, 0), (2, 2), (0, 2), (0, -2), (2, -2)],
    'f': [(0, 0), (-2, 0), (-2, -4), (-2, -2), (0, -2)],
    'g': [(0, 0), (-2, 0), (-2, -4), (0, -4), (0, -2), (-1, -2)],
    'h': [(0, 0), (0, -4), (0, -2), (2, -2), (2, -4)],
    'i': [(0, 0), (2, 0), (1, 0), (1, -4), (0, -4), (2, -4)],
    'j': [(0, 0), (2, 0), (2, -4), (0, -4), (0, -2)],
    'k': [(0, 0), (0, -4), (2, -2), (1, -3), (2, -4)],
    'l': [(0, 0), (0, -4), (2, -4)],
    'm': [(0, 0), (0, 4), (2, 2), (4, 4), (4, 0)],
    'n': [(0, 0), (0, 4), (2, 0), (2, 4)],
    'p': [(0, 0), (2, 0), (2, 2), (0, 2), (0, -2)],
    'q': [(0, 0), (-2, 0), (-2, 2), (0, 2), (0, -2), (0.2, -2)],
    's': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, -4), (-2, -4)],
    't': [(0, 0), (2, 0), (1, 0), (1, -4)],
    'u': [(0, 0), (0, -2), (2, -2), (2, 0)],
    'r': [(0, 0), (2, 0), (2, 2), (0, 2), (0, -2), (0,0), (2,-2)],
    'v': [(0, 0), (2, -4), (4, 0)],
    'w': [(0, 0), (0, -4), (2, -2), (4, -4), (4, 0)],
    'x': [(0, 0), (2, -4), (1, -2), (2, 0), (0, -4)], 
    'y': [(0, 0), (2, -2), (4, 0), (0, -4)],
    'z': [(0, 0), (2, 0), (0, -2), (2, -2)]
}

# Function to calculate direction and distance
def calculate_direction_and_time(start, end, speed=1.5):
    max_distance = 25  # Maximum comfortable distance for movement
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = np.sqrt(dx**2 + dy**2)

    # If the distance exceeds max_distance, adjust the time to cap it
    if distance > max_distance:
        time_required = max_distance / speed
    else:
        time_required = distance / speed

    vibration_intensity = 50
    
    if dx > 0 and dy == 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=120,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'right', time_required
    elif dx < 0 and dy == 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=45,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'left', time_required
    elif dy > 0 and dx == 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=90,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )     
            return 'top', time_required
    elif dy < 0 and dx == 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=60,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'down', time_required
    elif dx > 0 and dy > 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b110000,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'diagonal right top', time_required
    elif dx > 0 and dy < 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b101000,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'diagonal right bottom', time_required
    elif dx < 0 and dy > 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b010100,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'diagonal left top', time_required
    elif dx < 0 and dy < 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b001100,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'diagonal left bottom', time_required
    else:
        return 'none', 0
    

# Function to simulate tactile feedback based on shape
def simulate_tactile_feedback(shape, speed=1.5):
    vertices = shapes[shape]
    vertices.append(vertices[-1])  # Add the last vertex again to complete the shape

    for i in range(len(vertices) - 1):
        start = vertices[i]
        end = vertices[i + 1]
        direction, time_required = calculate_direction_and_time(start, end, speed)
        if direction != 'none':
            print(f"{direction} for {time_required:.2f} seconds")
            time.sleep(time_required) # Simulate the time required for the movement
            belt_controller.stop_vibration()
            time.sleep(1)


# Define the categories and their items
categories = {
    'numbers': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'letters': ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'l', 'p', 'q', 'u'],
    'beta': ['k', 'm', 'n', 'r', 'v', 'w', 'y', 'z']
}

# Shuffle the items within each category for each participant
for category, items in categories.items():
    random.shuffle(items)

# Execute drawing tasks for each category sequentially
for category, items in categories.items():
    print(f"Starting category: {category}\n")
    for index, item in enumerate(items):
        time.sleep(3)
        print(item)
        simulate_tactile_feedback(item)
        print("stop \n")
        if belt_controller:
            belt_controller.stop_vibration()
            belt_controller.send_pulse_command(
                channel_index=0,
                orientation_type=BeltOrientationType.BINARY_MASK,
                orientation=0b111100,
                intensity=100,
                on_duration_ms=150,
                pulse_period=500,
                pulse_iterations=5,
                series_period=5000,
                series_iterations=1,
                timer_option=BeltVibrationTimerOption.RESET_TIMER,
                exclusive_channel=False,
                clear_other_channels=False)
        time.sleep(5)  # Pause after each shape

        # Add a 5-second rest after every 5 items within the category
        #if (index + 1) % 5 == 0:
        #    print("5-second rest \n")
        #    time.sleep(5)

belt_controller.disconnect_belt() if belt_controller else None
sys.exit()