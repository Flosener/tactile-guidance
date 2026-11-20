# Code modified from original created by Ramon Zacharias
# Original code from RZ can be found at https://github.com/R3yman/TactileBraceletExperiment

# Import necessary libraries and modules
import csv
import random
import time
from pathlib import Path
from random import randrange

import keyboard
import pygame
from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption)

from connect import interactive_belt_connect, setup_logger

import os.path   
import numpy as np


# a delegate class for the BeltController
class Delegate(BeltControllerDelegate):
    # Belt controller delegate
    pass

# Initialize pygame, set random seed, and define Global variables
pygame.init()
random.seed()

# Global variables
participantID = 0
belt_controller_delegate = Delegate()
belt_controller = BeltController(belt_controller_delegate)
obs_localization = []
block_localization = 1
presented_stimuli_localization = []
obs_grasping_t = [["time", "num_instructions", "location", "block", "condition", "success"]]
obs_grasping_a = [["time", "num_instructions", "location", "block", "condition", "success"]]
vibration_intensity = 100

from pynput import keyboard as kb

_pressed = set()

def _on_press(key):
    _pressed.add(key)

def _on_release(key):
    _pressed.discard(key)

kb.Listener(on_press=_on_press, on_release=_on_release, daemon=True).start()

def is_pressed(name):
    if len(name) == 1:
        return kb.KeyCode.from_char(name) in _pressed
    return getattr(kb.Key, name) in _pressed   # "enter", "space", "esc", "up", ...


# the main function
def main():

    print("Welcome to the \"Augmenting functional vision using tactile guidance\" experiment!")
    print("Please connect the tactile bracelet and press Enter to continue.")

    # Wait for user to continue.
    while True:
        if is_pressed("enter"):
            break

    belt_controller = connect_belt()

    participantID = ""
    while participantID == "":
        participantID = get_input("Please enter a participant ID: ")
        if participantID == "":
            print("No ID entered. Please enter an ID.")
        else:
            print("Participant ID: " + participantID)
            continue

    # for tactile
    global targets_t

    if os.path.isfile(f'{participantID}_targets_t.csv'):
        targets_t = np.loadtxt(f'{participantID}_targets_t.csv', delimiter=',', dtype=int)
        
    else:
        # list of numbers repeated 8 times
        numbers_t = [i for i in range(1, 10)] * 8
        # shuffle the list
        random.shuffle(numbers_t)
        # split the shuffled list into 8 block 
        targets_t = [numbers_t[i:i+9] for i in range(0, len(numbers_t), 9)]
        np.savetxt(f'{participantID}_targets_t.csv', targets_t, delimiter=',', fmt="%d")  

    # for auditory
    global targets_a
    
    if os.path.isfile(f'{participantID}_targets_a.csv'):
        targets_a = np.loadtxt(f'{participantID}_targets_a.csv', delimiter=',', dtype=int)

    else:
        # list of numbers repeated 8 times
        numbers_a = [i for i in range(1, 10)] * 8
        # shuffle the list
        random.shuffle(numbers_a)
        # split the shuffled list into 8 block 
        targets_a = [numbers_a[i:i+9] for i in range(0, len(numbers_a), 9)]
        np.savetxt(f'{participantID}_targets_a.csv', targets_a, delimiter=',', fmt="%d")

    coinflip = random.sample([0,1], 1)[0]

    if coinflip == 0:
        print("Condition: Tactile")
    elif coinflip == 1:
        print("Condition: Auditory")

    print("Press 1 to start the localization task. Press 2 to start the grasping task.")

    prev_block_auditory = 0
    prev_block_tactile = 0

# Select localization or grasping task.
    while True:
        user_in = get_input("Localization or grasping? [1,2]", type_=int, min_=1, max_=2)

        # Start localization task
        if user_in == 1:
            localization_task()
            print(obs_localization)
            calc_accuracy()
            write_to_csv(participantID, obs_localization, "localization")
            write_to_csv(participantID, presented_stimuli_localization, "stimuli")

        # Start grasping task
        if user_in == 2:
            user_in = get_input("Tactile or auditory condition?. Press 0 to return. [1,2,0]", type_=int, min_=0, max_=2)
            if user_in == 1:
                prev_block_tactile = grasping_task("tactile",prev_block_tactile)
                write_to_csv(participantID, obs_grasping_t, "grasping_t")
            elif user_in == 2:
                print("Auditory condition selected. You will guide the participant to the correct fruit verbally.")
                prev_block_auditory = grasping_task("auditory",prev_block_auditory)
                write_to_csv(participantID, obs_grasping_a, "grasping_a")
            elif user_in == 0:
                continue


# the localization task
def localization_task():
    while belt_controller.get_connection_state() == BeltConnectionState.CONNECTED:
        print("Q. to quit.")
        print("0. Stop vibration.")
        print("1. Example stimuli.")
        print("2. Start block.")
        
        global block_localization
       
        while True:
            user_in = get_input("[Q,0,1,2]: ", type_=str, range_=["0", "1", "2", "Q","q"])
            
            # Start a block.
            if user_in == "2":
                stop = False
                while not stop:
                    print("Current block is number " + str(block_localization))
                    stimuli = [45, 45, 45, 45,
                               60, 60, 60, 60,
                               90, 90, 90, 90,
                               120, 120, 120, 120]  # List of orientations

                    random.shuffle(stimuli)  # Shuffle orientations
                    obs_localization.append([])
                    trans_stimuli = []  # List with stimuli translated from orientation to direction.
                    # Translate stimuli from orientation to direction.
                    for stimulus in stimuli:
                        if stimulus == 45: trans_stimuli.append("left")
                        elif stimulus == 60: trans_stimuli.append("down")
                        elif stimulus == 90: trans_stimuli.append("up")
                        elif stimulus == 120: trans_stimuli.append("right")
                    presented_stimuli_localization.append(trans_stimuli)

                    time.sleep(3)
                    for stimulus in stimuli:
                        belt_controller.send_pulse_command(
                            channel_index=0,
                            orientation_type=BeltOrientationType.ANGLE,
                            orientation=stimulus,
                            intensity=vibration_intensity,
                            on_duration_ms=500,
                            pulse_period=2000,
                            pulse_iterations=1,
                            series_period=1500,
                            series_iterations=1,
                            timer_option=BeltVibrationTimerOption.RESET_TIMER,
                            exclusive_channel=False,
                            clear_other_channels=False
                        )
                        obs_localization[block_localization-1].append(collect_response())

                    user_in = get_input("Block " + str(block_localization) + " completed. Continue?[y,n]", type_=str, range_=["y","n"])
                    block_localization += 1
                    if user_in == "n": stop = True 
                print(presented_stimuli_localization)
                break

            # Present example stimuli.
            elif user_in == "1":
                # Up
                belt_controller.send_pulse_command(
                    channel_index=0,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=90,
                    intensity=vibration_intensity,
                    on_duration_ms=500,
                    pulse_period=2000,
                    pulse_iterations=1,
                    series_period=1500,
                    series_iterations=1,
                    timer_option=BeltVibrationTimerOption.RESET_TIMER,
                    exclusive_channel=False,
                    clear_other_channels=False
                )
                time.sleep(3)
                # Right
                belt_controller.send_pulse_command(
                    channel_index=0,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=120,
                    intensity=vibration_intensity,
                    on_duration_ms=500,
                    pulse_period=1000,
                    pulse_iterations=1,
                    series_period=1500,
                    series_iterations=1,
                    timer_option=BeltVibrationTimerOption.RESET_TIMER,
                    exclusive_channel=False,
                    clear_other_channels=False
                )
                time.sleep(3)
                # Down
                belt_controller.send_pulse_command(
                    channel_index=0,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=60,
                    intensity=vibration_intensity,
                    on_duration_ms=500,
                    pulse_period=1000,
                    pulse_iterations=1,
                    series_period=1500,
                    series_iterations=1,
                    timer_option=BeltVibrationTimerOption.RESET_TIMER,
                    exclusive_channel=False,
                    clear_other_channels=False
                )
                time.sleep(3)
                # Left
                belt_controller.send_pulse_command(
                    channel_index=0,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=45,
                    intensity=vibration_intensity,
                    on_duration_ms=500,
                    pulse_period=1000,
                    pulse_iterations=1,
                    series_period=1500,
                    series_iterations=1,
                    timer_option=BeltVibrationTimerOption.RESET_TIMER,
                    exclusive_channel=False,
                    clear_other_channels=False
                )
                time.sleep(3)
                break
            # Quit the task.
            if user_in.lower() == "q":
                return


# the grasping task
def grasping_task(condition, prev):
    new_trial = True
    num_instructions = 0
    time_limit = 20
    curr = ""
    last = ""
    begin = 0
    target_idx = 0
    rep_idx = 0
    rep_list = []
    time_limit_reached = False

    # Present example stimuli.
    if condition == "tactile":
        user_in = get_input("Present example stimuli? [y,n]", range_=("y", "n"))
        if user_in == "y":
            present_example_stimuli("tactile")

    # Get block number or quit.
    if condition == "tactile":
        print("The previous block number was " + str(prev))
        block_tactile = get_input("Enter a block number or press 0 to quit the task.", type_=int, min_=0)
        prev = block_tactile
        target_list = targets_t[block_tactile - 1]
        if block_tactile == 0:  # Quit if user enters 0.
            return
        
    if condition == "auditory":
        print("The previous block number was " + str(prev))
        block_auditory = get_input("Enter a block number or press 0 to quit the task.", type_=int, min_=0)
        prev = block_auditory
        target_list = targets_a[block_auditory -1]
        if block_auditory == 0:  # Quit if user enters 0.
            return
    
    # Print targets.
    print("Target order: " + str(target_list))
    print("Target: " + str(target_list[target_idx]))

    while True:
        # Check for start of trial.
        if (is_pressed("left") or is_pressed("right") or is_pressed("up") or is_pressed("down") or is_pressed("g")) and new_trial and not is_pressed("s"):
            begin = time.perf_counter()
            new_trial = False
        
        # Quit the task.
        if is_pressed("q"):
            if condition == "tactile":
                belt_controller.stop_vibration()
            return
        
        # Check time limit.
        if begin != 0:
            if(time.perf_counter() - begin > time_limit) and not new_trial:
                time_limit_reached = True
        
        # Stop the trial, calculate the time and append the observations.
        if (is_pressed("s") or (time_limit_reached)) and not new_trial:
            if condition == "tactile":
                belt_controller.stop_vibration()

            if time_limit_reached:
                elapsed = time_limit
                result = "fail"
                print("Time limit reached.")

            if not time_limit_reached:
                end = time.perf_counter()
                elapsed = end - begin # Calculate trial time.
                print("Trial completed.")
                print("Completion time is ", elapsed, "seconds.")
                result = get_input("Press s for success, d for fail or f for experimenter fail.[s,d,f]", type_=str, range_=("s", "d", "f"))
                if result == "s": result = "success"
                elif result == "d": result = "fail"
                # Experimenter fail, append target to repetition list.
                else: 
                    result = "exFail"
                    if target_idx < len(target_list):  # Check if all targets have been recorded before.
                        rep_list.append(target_list[target_idx])  # Append target where experimenter failed to repetition list.
                    else: # If all targets have been recorded.
                        rep_list.append(rep_list[rep_idx])

            # No experimenter fail.
            if target_idx < len(target_list): 
                if condition == "tactile": 
                    obs_grasping_t.append([elapsed, num_instructions, target_list[target_idx], block_tactile, "tactile", result])
                elif condition == "auditory":
                    obs_grasping_a.append([elapsed, num_instructions, target_list[target_idx], block_auditory, "auditory", result])

            elif len(rep_list) > rep_idx:
                if condition == "tactile":
                    obs_grasping_t.append([elapsed, num_instructions, "rep_" + str(rep_list[rep_idx]), block_tactile, "tactile", result])
                    rep_idx += 1
                elif condition == "auditory":
                    obs_grasping_a.append([elapsed, num_instructions, "rep_" + str(rep_list[rep_idx]), block_auditory, "auditory", result])
                    rep_idx += 1

            num_instructions = 0
            last = ""
            new_trial = True
            time_limit_reached = False
            target_idx += 1
            # user_in

            if target_idx > len(target_list)-1:  # Check if all targets have been recorded.
                if len(rep_list) != 0: # Check if targets must be repeated.
                    if len(rep_list) > rep_idx:
                        print("Repetition target:" + str(rep_list[rep_idx]))
                    else: 
                        print("Block finished!")
                        return prev
                else: 
                    print("Block finished!")
                    return prev
            else: print(target_list[target_idx]) # Print target.

        elif is_pressed('right') and not new_trial:
            curr = "r"
            if curr != last:
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(120, channel_index=0, intensity=vibration_intensity)
                num_instructions += 1
                print("Right")
                last = curr

        elif is_pressed('left') and not new_trial:
            curr = "l"
            if curr != last:
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(45, channel_index=0, intensity=vibration_intensity)
                num_instructions += 1
                print("Left")
                last = curr


        elif is_pressed('down') and not new_trial:
            curr = "d"
            if curr != last:
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(60, channel_index=0, intensity=vibration_intensity)
                num_instructions += 1
                print("Down")
                last = curr

        elif is_pressed('up') and not new_trial:
            curr = "u"
            if last != curr:
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(90, channel_index=0, intensity=vibration_intensity)
                num_instructions += 1
                print("Up")
                last = curr

        elif is_pressed('G') and not new_trial:
            curr = "f"
            if last != curr:
                if condition == "tactile":
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=vibration_intensity,
                        on_duration_ms=150,
                        pulse_period=500,
                        pulse_iterations=9,
                        series_period=5000,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                num_instructions += 1
                print("Grasp")
                last = curr

# the function to present example stimuli
def present_example_stimuli(condition):

    curr = ""
    last = ""

    sample_targets = random.sample(list(range(1,10)), 3) # Draw 3 random targets
    print("Example targets: " + str(sample_targets))

    while True:
        # Stop the stimulus.
        if is_pressed('s'):
            if condition == "tactile":
                belt_controller.stop_vibration()
            last = ""
        # Present right stimulus.
        elif is_pressed('right'):
            curr = "r"
            if curr != last:
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(120, channel_index=0, intensity=vibration_intensity)
                print("Right")
                last = curr
        # Present up stimulus.
        elif is_pressed('up'):
            curr = "u"
            if curr != last:
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(90, channel_index=0, intensity=vibration_intensity)
                print("Up")
                last = curr
        # Present down stimulus.
        elif is_pressed('down'):
            curr = "d"
            if curr != last:
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(60, channel_index=0, intensity=vibration_intensity)
                print("Down")
                last = curr
        # Present left stimulus.
        elif is_pressed('left'):
            curr = "l"
            if curr != last:
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(45, channel_index=0, intensity=vibration_intensity)
                print("Left")
                last = curr

        # Grasping Command
        elif is_pressed('g'):
            curr = "g"
            if curr != last:
                if condition == "tactile":
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=vibration_intensity,
                        on_duration_ms=150,
                        pulse_period=500,
                        pulse_iterations=9,
                        series_period=5000,
                        series_iterations=3,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                        )
                print("Grasp")
                last = curr

        # Quit.
        if is_pressed("q"):
            if condition == "tactile":
                belt_controller.stop_vibration()
            break


def write_to_csv(id, observations, task):

    # if file exists read rows into list so that new obs can be added to old
    id = id
    obs = observations
    if task == "localization":
        filepath = str(id + "_localization" + ".csv")

    elif task == "grasping_t":
        filepath = str(id + "_grasping_t" + ".csv")

    elif task == "grasping_a":
        filepath = str(id + "_grasping_a" + ".csv")
    
    elif task == "stimuli":
        filepath = str(id + "_stimuli" + ".csv")

    with open(filepath, 'a', newline="") as file:
        writer = csv.writer(file)
        for list in obs:
            writer.writerow(list)


def connect_belt():
    setup_logger()

    # Interactive script to connect the belt
    interactive_belt_connect(belt_controller)
    if belt_controller.get_connection_state() != BeltConnectionState.CONNECTED:
        print("Connection failed.")
        return 0

    # Change belt mode to APP mode
    belt_controller.set_belt_mode(BeltMode.APP_MODE)


def collect_response():
    time_post_stim = time.perf_counter()
    response = ""
    while time.perf_counter() - time_post_stim < 3:  # Listen to keyboard input for 3 seconds.
        if is_pressed("left"):
            response = "left"
        if is_pressed("right"):
            response = "right"
        if is_pressed("up"):
            response = "up"
        if is_pressed("down"):
            response = "down"
    if response == "":
        response = "no response"
    print(response)
    return response


def generate_participantID():
    participantID = str(str(randrange(10)) + str(randrange(10)) + str(randrange(10)) + str(randrange(10)))
    return participantID


# Function to get user input taken from stack overflow
def get_input(prompt, type_=None, min_=None, max_=None, range_=None):
    if min_ is not None and max_ is not None and max_ < min_:
        raise ValueError("min mus be less than or equal to max.")
    while True:
        ui = input(prompt)
        if type_ is not None:
            try:
                ui = type_(ui)
            except ValueError:
                print("Input type must be {0}.".format(type_.__name__))
                continue
        if max_ is not None and ui > max_:
            print("Input must be less than or equal to {0}.".format(max_))
        elif min_ is not None and ui < min_:
            print("Input must be greater than or equal to {0}.".format(min_))
        elif range_ is not None and ui not in range_:
            if isinstance(range_, range):
                template = "Input must be between {0.start} and {0.stop}."
                print(template.format(range_))
            else:
                template = "Input must be {0}."
                if len(range_) == 1:
                    print(template.format(*range_))
                else: 
                    expected = " or ".join((
                        ", ".join(str(x) for x in range_[:-1]),
                        str(range_[-1])
                    ))
                    print(template.format(expected))
        else:
            return ui


def calc_accuracy():
    correct = 0
    num_blocks = 0
    mean_correct = 0

    # accuracy for first block
    if len(obs_localization) == 1:  # Check if at least 3 blocks were recorded
        for i in range(len(obs_localization[0])):    
            if obs_localization[0][i] == presented_stimuli_localization[0][i]:
                correct += 1 
        num_blocks += 1

        if num_blocks > 0:
            mean_correct = correct / (num_blocks * len(obs_localization[0]))
    
        print(str(mean_correct) + "in 1st block")  

    # accuracy for second block
    if len(obs_localization) == 2: # Check if at least 3 blocks were recorded
        for i in range(len(obs_localization[1])):    
            if obs_localization[1][i] == presented_stimuli_localization[1][i]:
                correct += 1 
        num_blocks += 1

        if num_blocks > 0:
            mean_correct = correct / (num_blocks * len(obs_localization[0]))
        print(str(mean_correct) + "in 2nd block")

    if len(obs_localization) >= 3:  # Check if at least 3 blocks were recorded
        for x in range(-3,0):  # Go through the last 3 blocks
            for i in range(len(obs_localization[x])):    
                if obs_localization[x][i] == presented_stimuli_localization[x][i]:
                    correct += 1 
            num_blocks += 1

        if num_blocks > 0:
            mean_correct = correct / (num_blocks * len(obs_localization[0]))
    
        print(str(mean_correct) + "in " + str(num_blocks) + " blocks")


if __name__ == "__main__":
    main()