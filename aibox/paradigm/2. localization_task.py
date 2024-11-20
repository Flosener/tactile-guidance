# System
import sys
import os

# Use the project file packages instead of the conda packages, i.e. add to system path for import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import keyboard
import time
import json
import numpy as np
import random

from pybelt.belt_controller import (BeltOrientationType, BeltVibrationPattern)

from bracelet import connect_belt, BraceletController
from controller import close_app

# Define shapes with vertices
general_directions = {
    'left': [(0, 0), (-4, 0)],
    'right': [(0, 0), (4, 0)],
    'up': [(0, 0), (0, 4)],
    'down': [(0, 0), (0, -4)]}

diagonal_directions = {
    'diagonal_tr': [(0, 0), (3, 3)],
    'diagonal_tl': [(0, 0), (-3, 3)],
    'diagonal_br': [(0, 0), (3, -3)],
    'diagonal_bl': [(0, 0), (-3, -3)]
}

advanced_shapes = {'square': [(0,0), (0,1), (1,1), (1,0), (0,0)],
                   'triangle': [(0,0), (1,2), (2,0), (0,0)]}

"""advanced_shapes = {'0': [(0, 0), (0, 4), (2, 4), (2, 0), (0, 0)],
    '1': [(0, 0), (0, -2)],
    '2': [(0, 0), (2, 0), (2, -2), (0, -2), (0, -4), (2, -4)],
    '3': [(0, 0), (2, 0), (2, -2), (0, -2), (2, -2), (2, -4), (0, -4)],
    '4': [(0, 0), (0, -2), (2, -2), (2, 0), (2, -4)],
    '5': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, -4), (-2, -4)],
    '6': [(0, 0), (-2, 0), (-2, -4), (0, -4), (0, -2), (-2, -2)],
    '7': [(0, 0), (2, 0), (2, -4)],
    '8': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, -4), (-2, -4), (-2, -2), (0, -2), (0, 0)],
    '9': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, 0), (0, -4), (-2, -4)],
    'a': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, 0), (0, -2.5)],
    'b': [(0, 0), (0, -4), (2, -4), (2, -2), (0, -2)],
    'c': [(0, 0), (-2, 0), (-2, -2), (0, -2)],
    'd': [(0, 0), (0, -4), (-2, -4), (-2, -2), (0, -2)],
    'e': [(0, 0), (2, 0), (2, 2), (0, 2), (0, -2), (2, -2)],
    'f': [(0, 0), (-2, 0), (-2, -4), (-2, -2), (0, -2)],
    'h': [(0, 0), (0, -4), (0, -2), (2, -2), (2, -4)],
    'i': [(0, 0), (4, 0), (2, 0), (2, -4), (0, -4), (4, -4)],
    'j': [(0, 0), (2, 0), (2, -4), (0, -4), (0, -2)],
    'k': [(0, 0), (0, -4), (2, -2), (1, -3), (2, -4)],
    'l': [(0, 0), (0, -4), (2, -4)],
    'm': [(0, 0), (0, 4), (2, 2), (4, 4), (4, 0)],
    'n': [(0, 0), (0, 4), (2, 0), (2, 4)],
    'p': [(0, 0), (0, 4), (2, 4), (2, 2), (0, 2)],
    'q': [(0, 0), (0, 4), (-2, 4), (-2, 2), (0, 2)],
    'u': [(0, 0), (0, -2), (2, -2), (2, 0)],
    'r': [(0, 0), (0, 4), (2, 4), (2, 2), (0, 2), (2, 0)],
    'v': [(0, 0), (2, -4), (4, 0)],
    'w': [(0, 0), (0, -4), (2, -2), (4, -4), (4, 0)],
    'y': [(0, 0), (2, -2), (4, 0), (0, -4)],
    'z': [(0, 0), (2, 0), (0, -2), (2, -2)]}"""

# Define the categories and their items
categories = {
    'directions': ['left', 'right', 'up', 'down', 'diagonal_1', 'diagonal_2', 'diagonal_3', 'diagonal_4'],
    'numbers': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'letters': ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'l', 'p', 'q', 'u'],
    'beta': ['k', 'm', 'n', 'r', 'v', 'w', 'y', 'z']
}

def calculate_direction_and_time(vibration_intensities, start, end, speed=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = np.sqrt(dx**2 + dy**2)
    time_required = distance / speed 

    bottom_intensity, top_intensity, left_intensity, right_intensity = vibration_intensities["bottom"], vibration_intensities["top"], vibration_intensities["left"], vibration_intensities["right"]

    #vibration_intensity = 50
    
    orientation_mapping = {"bottom": 60,
                           "top": 90,
                           "left": 120,
                           "right": 45}

    if dx > 0 and dy == 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=right_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['right'],
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
            intensity=left_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['left'],
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
            intensity=top_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['top'],
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
            intensity=bottom_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['bottom'],
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
            intensity=right_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['right'],
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            belt_controller.send_vibration_command(
            channel_index=1,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=top_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['top'],
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
            intensity=right_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['right'],
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            belt_controller.send_vibration_command(
            channel_index=1,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=bottom_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['bottom'],
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
            intensity=left_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['left'],
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            belt_controller.send_vibration_command(
            channel_index=1,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=top_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['top'],
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
            intensity=left_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['left'],
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            belt_controller.send_vibration_command(
            channel_index=1,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=bottom_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation_mapping['bottom'],
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'diagonal left bottom', time_required
    else:
        return 'none', 0
    
def draw_shape(vibration_intensities, shape, speed=1):
    vertices = shapes[shape]
    vertices.append(vertices[-1])  # Add the last vertex again to complete the shape

    for i in range(len(vertices) - 1):
        start = vertices[i]
        end = vertices[i + 1]
        direction, time_required = calculate_direction_and_time(vibration_intensities, start, end, speed)
        if direction != 'none':
            print(f"{direction} for {time_required:.2f} seconds")
            time.sleep(time_required) # Simulate the time required for the movement
            if belt_controller:
                belt_controller.stop_vibration()
                time.sleep(1)

def localization_task(categories, vibration_intensities):
    
    # Shuffle the items within each category for each participant
    for category, items in categories.items():
        random.shuffle(items)

    for category, items in categories.items():
        for index, item in enumerate(items):
            time.sleep(3)
            print(item)
            draw_shape(vibration_intensities, item)
            print("stop \n")

def draw_shapes_bracelet(belt_controller, bracelet, shapes, randomize_order=False):

    if randomize_order:
        random.shuffle(shapes)

    for shape in shapes:
        print(shape)
        vertices = shapes[shape]
        #vertices = vertices +  vertices[0]
        print(vertices)

        for i in range(len(vertices) - 1):    

            print(f'start: {vertices[i]}, end: {vertices[i+1]}')
            right_int, left_int, bot_int, top_int, _ = bracelet.get_intensity(vertices[i], vertices[i+1], bracelet.vibration_intensities, None) # mirroring left and right intensities as well as top and bottom
            print(f'intensities: R {right_int}, L {left_int}, T {top_int}, B {bot_int}')

            trial_start_time = time.time()

            belt_controller.send_vibration_command(
                channel_index=0,
                pattern=BeltVibrationPattern.CONTINUOUS,
                intensity=right_int,
                orientation_type=BeltOrientationType.ANGLE,
                orientation=120,
                pattern_iterations=None,
                pattern_period=500,
                pattern_start_time=0,
                exclusive_channel=False,
                clear_other_channels=False
            )
            belt_controller.send_vibration_command(
                channel_index=1,
                pattern=BeltVibrationPattern.CONTINUOUS,
                intensity=left_int,
                orientation_type=BeltOrientationType.ANGLE,
                orientation=45,
                pattern_iterations=None,
                pattern_period=500,
                pattern_start_time=0,
                exclusive_channel=False,
                clear_other_channels=False
            )
            belt_controller.send_vibration_command(
                channel_index=2,
                pattern=BeltVibrationPattern.CONTINUOUS,
                intensity=top_int,
                orientation_type=BeltOrientationType.ANGLE,
                orientation=90,
                pattern_iterations=None,
                pattern_period=500,
                pattern_start_time=0,
                exclusive_channel=False,
                clear_other_channels=False
            )
            belt_controller.send_vibration_command(
                channel_index=3,
                pattern=BeltVibrationPattern.CONTINUOUS,
                intensity=bot_int,
                orientation_type=BeltOrientationType.ANGLE,
                orientation=60,
                pattern_iterations=None,
                pattern_period=500,
                pattern_start_time=0,
                exclusive_channel=False,
                clear_other_channels=False
            )
            
            while True:
                
                if time.time() > trial_start_time + 2:
                    belt_controller.stop_vibration()
                    break

            time.sleep(1)

    belt_controller.stop_vibration()


if __name__ == '__main__':
    participant = 1
    output_path = str(parent_dir) + '/results/'

    # load participants intensities frob bracelet calibration
    try:
        with open(output_path + f"calibration_participant_{participant}.json") as file:
            participant_vibration_intensities = json.load(file)
        print('Calibration intensities loaded succesfully.')
    except:
        while True:
            baseline_intensity = 50
            continue_with_baseline = input(f'Error while loading the calibration file. Do you want to continue with baseline intensity of {baseline_intensity} for each vibromotor? (y/n)')
            if continue_with_baseline == 'y':
                participant_vibration_intensities = {'bottom': baseline_intensity,
                                                     'top': baseline_intensity,
                                                     'left': baseline_intensity,
                                                     'right': baseline_intensity}
                break
            elif continue_with_baseline == 'n':
                sys.exit()

    assert len(participant_vibration_intensities) == 4, 'Participation intensities file is corrupted. Run bracelet_calibration again.'

    connection_check, belt_controller = connect_belt()
    if connection_check:
        print('Bracelet connection successful.')
    else:
        while True:
            continue_without_bracelet = input('Error connecting bracelet. Do you want to continue with print commands only? (y/n)')
            if continue_without_bracelet == 'y':
                break
            elif continue_without_bracelet == 'n':
                sys.exit()

    try:
        #localization_task(categories, participant_vibration_intensities)
        bracelet_controller = BraceletController(vibration_intensities=participant_vibration_intensities)

        while True:
            condition = input('Select condition: g - general directions, d - diagonals, s - shapes:')
            if condition == 'g':
                draw_shapes_bracelet(belt_controller, bracelet_controller, general_directions)
            elif condition == 'd':
                draw_shapes_bracelet(belt_controller, bracelet_controller, diagonal_directions)
            elif condition == 's':
                draw_shapes_bracelet(belt_controller, bracelet_controller, advanced_shapes)

    except KeyboardInterrupt:
        close_app(belt_controller)
    
    # In the end, kill everything
    if connection_check:
        close_app(belt_controller)