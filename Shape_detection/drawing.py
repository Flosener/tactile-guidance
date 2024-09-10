import numpy as np
import time
import sys
import random
import keyboard
from auto_connect import interactive_belt_connect, setup_logger
from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption, BeltVibrationPattern)
from bracelet import connect_belt
import threading
from threading import Event, Thread
import os

# Define shapes with vertices
shapes = {
    'cross': [(0, 0), (2, 0), (2, 2), (4, 2), (4, 0), (6, 0), (6, -2), (4, -2), (4, -4), (2, -4), (2, -2), (0, -2), (0, 0)],
    'square': [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)],
    'octagon': [(0, 0), (2, 2), (4, 2), (6, 0), (6, -2), (4, -4), (2, -4), (0, -2), (0, 0)],
    'star': [(0, 0), (3, 5), (6, 0), (0, 3), (6, 3), (0, 0)],
    '0': [(0, 0), (2, 0), (2, -4), (0, -4),  (0, 0), (0, 0)],
    '1': [(0, 0), (0, -2)],
    '2': [(0, 0), (2, 0), (2, -2), (0, -2), (0, -4), (2, -4)],
    '3': [(0, 0), (2, 0), (2, -2), (0, -2), (2, -2), (2, -4), (0, -4)],
    '4': [(0, 0), (0, -2), (2, -2), (2, 0), (2, -4)],
    '5': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, -4), (-2, -4)],
    '6': [(0, 0), (0, -4), (2, -4), (2, -2), (0, -2)],
    '7': [(0, 0), (2, 0), (2, -4)],
    '8': [(0, 0), (2, 0), (2, -2), (0, -2), (0, -4), (2, -4), (2, -2), (0, -2), (0, 0)],
    '9': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, 0), (0, -4), (-2, -4)],
    'a': [(0, 0), (-2, 0), (-2, 2), (0, 2), (0, -0.2), (0.2, -0.2)],
    'b': [(0, 0), (2, 0), (2, -2), (0, -2), (0, 2)],
    'c': [(0, 0), (-2, 0), (-2, -2), (0, -2)],
    'd': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, 2)],
    'e': [(0, 0), (2, 0), (2, 2), (0, 2), (0, -2), (2, -2)],
    'f': [(0, 0), (-2, 0), (-2, -4), (-2, -2), (0, -2)],
    'g': [(0, 0), (-2, 0), (-2, -4), (0, -4), (0, -2), (-1, -2)],
    'h': [(0, 0), (0, -4), (0, -2), (2, -2), (2, -4)],
    'i': [(0, 0), (2, 0), (1, 0), (1, -4), (0, -4), (2, -4)],
    'j': [(0, 0), (2, 0), (2, -4), (0, -4)],
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

def alpha():
    # Function to load calibration values
    def load_calibration_data(participant_ID):
        directory = r"C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/"
        #directory = r"D:/WWU/M8 - Master Thesis/Project/Code/"
        file_path = os.path.join(directory, f'intensity alpha_{participant_ID}.txt')
        
        calibration_data = {}
        
        # Read values from the text file
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(": ")
                # Convert value to integer if it's a number
                calibration_data[key] = int(value) if value.isdigit() else value
        
        return calibration_data
    
    def calculate_direction_and_time(start, end, int_top, int_left, int_bottom, int_right, avg_int, speed=1.5):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = np.sqrt(dx**2 + dy**2)
        time_required = distance / speed 

        # Handle simultaneous vibration for each direction
        if dx > 0 and dy == 0:
            if belt_controller:
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.CONTINUOUS,
                    intensity=int_right,
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
                    intensity=int_left,
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
                    intensity=int_top,
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
                    intensity=int_bottom,
                    orientation_type=BeltOrientationType.ANGLE,                        
                    orientation=60,
                    pattern_iterations=None,
                    pattern_period=500,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False
                )
            return 'down', time_required
        else:
            return 'none', 0

    # Function to simulate tactile feedback based on shape
    def simulate_tactile_feedback(shape, speed=1.5):
        vertices = shapes[shape]
        vertices.append(vertices[-1])  # Add the last vertex again to complete the shape

        for i in range(len(vertices) - 1):
            start = vertices[i]
            end = vertices[i + 1]
            direction, time_required = calculate_direction_and_time(start, end,  int_top, int_left, int_bottom, int_right, avg_int, speed)
            if direction != 'none':
                time.sleep(0.2)
                print(f"{direction} for {time_required:.2f} seconds")
                time.sleep(time_required) # Simulate the time required for the movement
                belt_controller.stop_vibration()
                time.sleep(1)

    # Function for drawing examples
    def draw_examples():
        examples = {
            'cardinal': ['square', 'cross'], 
        }
        
        for category, items in examples.items():
            print(f"\nStarting example category: {category}")
            for item in items:
                time.sleep(5)
                print(f'Start Now: \n{item}')
                time.sleep(1)
                simulate_tactile_feedback(item)
                print("stop \n")
                if belt_controller:
                    belt_controller.stop_vibration()
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.BINARY_MASK,
                        orientation=0b111100,
                        intensity=avg_int,
                        on_duration_ms=150,
                        pulse_period=500,
                        pulse_iterations=5,
                        series_period=5000,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False)
                time.sleep(5)  # 5 second pause after each shape

    def draw_categories():
        categories = {
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'letters': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'p', 'q', 's', 't', 'u'],
        }

        # Shuffle the items within each category for each participant
        for category, items in categories.items():
            random.shuffle(items)

        total_figures = 0

        # Open a file to save the drawing order
        directory = r"C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/"
        #directory = r"D:/WWU/M8 - Master Thesis/Project/Code/"
        file_path = f"{directory}drawing order alpha_{participant_ID}.txt"
        with open(file_path, 'w') as file:  
            # Execute drawing tasks for each category sequentially
            for category, items in categories.items():
                file.write(f"\nStarting category: {category}\n")
                print(f"\nStarting category: {category}")
                for index, item in enumerate(items):
                    file.write(f'{item}\n')  # Save each item to the file
                    time.sleep(3)
                    print(f'Start Now: \n{item}')
                    time.sleep(2)
                    simulate_tactile_feedback(item)
                    print("stop \n")
                    if belt_controller:
                        belt_controller.stop_vibration()
                        belt_controller.send_pulse_command(
                            channel_index=0,
                            orientation_type=BeltOrientationType.BINARY_MASK,
                            orientation=0b111100,
                            intensity=avg_int,
                            on_duration_ms=150,
                            pulse_period=500,
                            pulse_iterations=5,
                            series_period=5000,
                            series_iterations=1,
                            timer_option=BeltVibrationTimerOption.RESET_TIMER,
                            exclusive_channel=False,
                            clear_other_channels=False)
                    time.sleep(5)  # 5 second pause after each shape

                    # 30 seconds pause after every 10 figures                    
                    total_figures += 1
                    if total_figures % 10 == 0:
                        print("Taking 15 seconds pause")
                        time.sleep(15)

                    # Prompt the user to press 'Enter' to continue
                    print("Press 'Enter' to proceed to the next figure \n")
                    while True:
                        if keyboard.is_pressed('enter'):
                            break

    if __name__ == "__main__":
        # Load the calibration data
        calibration_data = load_calibration_data(participant_ID)
            
        # Access the values
        int_top = calibration_data['int_top']
        int_bottom = calibration_data['int_bottom']
        int_left = calibration_data['int_left']
        int_right = calibration_data['int_right']
        avg_int = calibration_data['avg_int']
        
        while True:
            # Display menu options
            print("\nChoose an option:")
            print("0: Finish")
            print("1: Drawing Examples")
            print("2: Drawing Categories")

            choice = input("\nEnter 0, 1, or 2: ")

            if choice == '0':
                print("Exiting")
                break
            elif choice == '1':
                draw_examples()
            elif choice == '2':
                draw_categories()
            else:
                print("Invalid choice. Please enter 0, 1, or 2.")


def beta():
    # Function to load calibration values
    def load_calibration_data(participant_ID):
        directory = r"C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/"
        #directory = r"D:/WWU/M8 - Master Thesis/Project/Code/"
        file_path = os.path.join(directory, f'intensity beta_{participant_ID}.txt')
        
        calibration_data = {}
        
        # Read values from the text file
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(": ")
                # Convert value to integer if it's a number
                calibration_data[key] = int(value) if value.isdigit() else value
        
        return calibration_data
    
    # Define stop_event globally
    stop_event = Event()

    def calculate_direction_and_time(start, end, speed=1.5):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = np.sqrt(dx**2 + dy**2)
        time_required = distance / speed

        if dx > 0 and dy == 0:
            direction = 'right'
        elif dx < 0 and dy == 0:
            direction = 'left'
        elif dy > 0 and dx == 0:
            direction = 'top'
        elif dy < 0 and dx == 0:
            direction = 'down'
        elif dx > 0 and dy > 0:
            direction = 'top right'
        elif dx > 0 and dy < 0:
            direction = 'bottom right'
        elif dx < 0 and dy > 0:
            direction = 'top left'
        elif dx < 0 and dy < 0:
            direction = 'bottom left'
        else:
            direction = 'none'
        
        return direction, time_required   
    
    def vibrate_direction(direction, stop_event, preference, int_top, int_bottom, int_right, int_left, avg_int):
        # Check if belt_controller is initialized
        if not belt_controller:
            print("Error: Belt controller not initialized.")
            return

        try:      
            if preference == 'Interval':
                while not stop_event.is_set():
                    if direction == 'top':
                        belt_controller.send_vibration_command(
                            channel_index=0,
                            pattern=BeltVibrationPattern.CONTINUOUS,
                            intensity=int_top,
                            orientation_type=BeltOrientationType.ANGLE,
                            orientation=90,  # Top
                            pattern_iterations=None,
                            pattern_period=500,
                            pattern_start_time=0,
                            exclusive_channel=False,
                            clear_other_channels=False)
                    elif direction == 'down':
                        belt_controller.send_vibration_command(
                            channel_index=0,
                            pattern=BeltVibrationPattern.CONTINUOUS,
                            intensity=int_bottom,
                            orientation_type=BeltOrientationType.ANGLE,
                            orientation=60,  # Down
                            pattern_iterations=None,
                            pattern_period=500,
                            pattern_start_time=0,
                            exclusive_channel=False,
                            clear_other_channels=False)
                    elif direction == 'right':
                        belt_controller.send_vibration_command(
                            channel_index=0,
                            pattern=BeltVibrationPattern.CONTINUOUS,
                            intensity=int_right,
                            orientation_type=BeltOrientationType.ANGLE,
                            orientation=120,  # Right
                            pattern_iterations=None,
                            pattern_period=500,
                            pattern_start_time=0,
                            exclusive_channel=False,
                            clear_other_channels=False)
                    elif direction == 'left':
                        belt_controller.send_vibration_command(
                            channel_index=0,
                            pattern=BeltVibrationPattern.CONTINUOUS,
                            intensity=int_left,
                            orientation_type=BeltOrientationType.ANGLE,
                            orientation=45,  # Left
                            pattern_iterations=None,
                            pattern_period=500,
                            pattern_start_time=0,
                            exclusive_channel=False,
                            clear_other_channels=False)
                    elif direction == 'top right':
                        while not stop_event.is_set():
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.SINGLE_SHORT,
                                intensity=int_top,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=90,  # Top
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            time.sleep(0.2)
                            belt_controller.send_vibration_command(
                                channel_index=1,
                                pattern=BeltVibrationPattern.SINGLE_SHORT,
                                intensity=int_right,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=120,  # Right
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            time.sleep(0.2)
                    elif direction == 'top left':
                        while not stop_event.is_set():
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.SINGLE_SHORT,
                                intensity=int_top,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=90,  # Top
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            time.sleep(0.2)
                            belt_controller.send_vibration_command(
                                channel_index=1,
                                pattern=BeltVibrationPattern.SINGLE_SHORT,
                                intensity=int_left,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=45,  # Left
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            time.sleep(0.2)
                    elif direction == 'bottom right':
                        while not stop_event.is_set():
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.SINGLE_SHORT,
                                intensity=int_bottom,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=60,  # Down
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            time.sleep(0.2)
                            belt_controller.send_vibration_command(
                                channel_index=1,
                                pattern=BeltVibrationPattern.SINGLE_SHORT,
                                intensity=int_right,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=120,  # Right
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            time.sleep(0.2)
                    elif direction == 'bottom left':
                        while not stop_event.is_set():
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.SINGLE_SHORT,
                                intensity=int_bottom,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=60,  # Down
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            time.sleep(0.2)
                            belt_controller.send_vibration_command(
                                channel_index=1,
                                pattern=BeltVibrationPattern.SINGLE_SHORT,
                                intensity=int_left,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=45,  # Left
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            time.sleep(0.2)
                    else:
                        print(f"Direction '{direction}' not recognized.")
                        break

            elif preference == 'Simultaneous':
                    while not stop_event.is_set():
                        if direction == 'top':
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_top,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=90,  # Top
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            if stop_event.is_set():
                                break
                        elif direction == 'down':
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_bottom,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=60,  # Down
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            if stop_event.is_set():
                                break
                        elif direction == 'right':
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_right,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=120,  # Right
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            if stop_event.is_set():
                                break
                        elif direction == 'left':
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_left,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=45,  # Left
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            if stop_event.is_set():
                                break
                        elif direction == 'top right':
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_top,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=90,  # Top
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            belt_controller.send_vibration_command(
                                channel_index=1,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_right,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=120,  # Right
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            if stop_event.is_set():
                                break
                        elif direction == 'bottom right':
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_bottom,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=60, 
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            belt_controller.send_vibration_command(
                                channel_index=1,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_right,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=120,  
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            if stop_event.is_set():
                                break
                        elif direction == 'top left':
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_top,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=90,  # Top
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            belt_controller.send_vibration_command(
                                channel_index=1,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_left,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=45,  
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            if stop_event.is_set():
                                break
                        elif direction == 'bottom left':
                            belt_controller.send_vibration_command(
                                channel_index=0,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_bottom,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=60, 
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            belt_controller.send_vibration_command(
                                channel_index=1,
                                pattern=BeltVibrationPattern.CONTINUOUS,
                                intensity=int_left,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=45, 
                                pattern_iterations=None,
                                pattern_period=500,
                                pattern_start_time=0,
                                exclusive_channel=False,
                                clear_other_channels=False)
                            if stop_event.is_set():
                                break
                        else:
                            print(f"Direction '{direction}' not recognized.")
                            return

        
        except Exception as e:
            print(f"Exception occurred: {e}")
        finally:
            # Ensure vibrations are stopped
            belt_controller.stop_vibration()


    # Function to simulate tactile feedback based on shape
    def simulate_tactile_feedback(shape, preference, speed=1.5):
        vertices = shapes[shape]
        vertices.append(vertices[-1])  # Add the last vertex again to complete the shape

        for i in range(len(vertices) - 1):
            start = vertices[i]
            end = vertices[i + 1]
            direction, time_required = calculate_direction_and_time(start, end, speed)
            
            if direction != 'none':
                # Create and start a new thread for vibration simulation
                stop_event = threading.Event()
                vibration_thread = threading.Thread(
                    target=vibrate_direction,
                    args=(direction, stop_event, preference, int_top, int_bottom, int_right, int_left, avg_int)
                )
                vibration_thread.start()
                
                # Wait for the required time while the thread simulates vibration
                print(f"{direction} for {time_required:.2f} seconds")
                time.sleep(time_required)
                
                # Stop the vibration
                stop_event.set()
                vibration_thread.join()  # Wait for the vibration thread to finish
                
                # Ensure there is a pause before starting the next segment
                time.sleep(1)

    # Function for drawing examples
    def draw_examples():
        examples = {
            'cardinal': ['square', 'cross'], 
            'oblique' : ['octagon', 'star']
        }
        
        for category, items in examples.items():
            print(f"\nStarting example category: {category}")
            for item in items:
                time.sleep(5)
                print(f'Start Now: \n{item}')
                time.sleep(1)
                simulate_tactile_feedback(item, preference)
                print("stop \n")
                if belt_controller:
                    belt_controller.stop_vibration()
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.BINARY_MASK,
                        orientation=0b111100,
                        intensity=avg_int,
                        on_duration_ms=150,
                        pulse_period=500,
                        pulse_iterations=5,
                        series_period=5000,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False)
                time.sleep(5)  # 5 second pause after each shape

    def draw_categories():
        categories = {
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'letters': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'p', 'q', 's', 't', 'u'],
            'beta': ['k', 'm', 'n', 'r', 'v', 'w', 'x', 'y', 'z']
        }

        # Shuffle the items within each category for each participant
        for category, items in categories.items():
            random.shuffle(items)

        total_figures = 0

        # Open a file to save the drawing order
        directory = r"C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/"
        #directory = r"D:/WWU/M8 - Master Thesis/Project/Code/"
        file_path = f"{directory}drawing order beta_{participant_ID}.txt"
        with open(file_path, 'w') as file:  
            # Execute drawing tasks for each category sequentially
            for category, items in categories.items():
                file.write(f"\nStarting category: {category}\n")
                print(f"\nStarting category: {category}")
                for index, item in enumerate(items):
                    file.write(f'{item}\n')  # Save each item to the file
                    time.sleep(3)
                    print(f'Start Now: \n{item}')
                    time.sleep(2)
                    simulate_tactile_feedback(item, preference)
                    print("stop \n")
                    if belt_controller:
                        belt_controller.stop_vibration()
                        belt_controller.send_pulse_command(
                            channel_index=0,
                            orientation_type=BeltOrientationType.BINARY_MASK,
                            orientation=0b111100,
                            intensity=avg_int,
                            on_duration_ms=150,
                            pulse_period=500,
                            pulse_iterations=5,
                            series_period=5000,
                            series_iterations=1,
                            timer_option=BeltVibrationTimerOption.RESET_TIMER,
                            exclusive_channel=False,
                            clear_other_channels=False)
                    time.sleep(5)  # 5 second pause after each shape

                    # 30 seconds pause after every 10 figures                    
                    total_figures += 1
                    if total_figures % 10 == 0:
                        print("Taking 15 seconds pause")
                        time.sleep(15)

                    # Prompt the user to press 'Enter' to continue
                    print("Press 'Enter' to proceed to the next figure \n")
                    while True:
                        if keyboard.is_pressed('enter'):
                            break

    if __name__ == "__main__":
        # Load the calibration data
        calibration_data = load_calibration_data(participant_ID)
            
        # Access the values
        preference = calibration_data['preference']
        int_top = calibration_data['int_top']
        int_bottom = calibration_data['int_bottom']
        int_left = calibration_data['int_left']
        int_right = calibration_data['int_right']
        avg_int = calibration_data['avg_int']

        while True:
            # Display menu options
            print("\nChoose an option:")
            print("0: Finish")
            print("1: Drawing Examples")
            print("2: Drawing Categories")

            choice = input("\nEnter 0, 1, or 2: ")

            if choice == '0':
                print("Exiting")
                break
            elif choice == '1':
                draw_examples()
            elif choice == '2':
                draw_categories()
            else:
                print("Invalid choice. Please enter 0, 1, or 2.")

if __name__ == "__main__":
    # Connect to the belt
    connection_check, belt_controller = connect_belt()
    if connection_check:
        print('Bracelet connection successful.')
    else:
        print('Error connecting bracelet. Aborting.')
        sys.exit()
        
    participant_ID = input("Enter Participant ID: ")
        
    while True:
        # Display menu options
        print("\nChoose an option:")
        print("0: Finish")
        print("1: Alpha")
        print("2: Beta")

        choice = input("\nEnter 0, 1, or 2: ")

        if choice == '0':
            print("Exiting")
            break            
        elif choice == '1':
            alpha()
        elif choice == '2':
            beta()
        else:        
            print("Invalid choice. Please enter 0, 1, or 2.")

    belt_controller.disconnect_belt() if belt_controller else None
    sys.exit()

