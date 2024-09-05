import numpy as np
import os
import random
import time
import sys
import pandas as pd
import keyboard
import threading
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from auto_connect import interactive_belt_connect, setup_logger
from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption, BeltVibrationPattern)
from pybelt.belt_scanner import BeltScanner
from openpyxl.workbook import Workbook
from bracelet import connect_belt

connection_check, belt_controller = connect_belt()
if connection_check:
    print('Bracelet connection successful.')
else:
    print('Error connecting bracelet. Aborting.')
    sys.exit()

# Define the event for stopping vibration
stop_event = threading.Event()

def esc_key_listener(stop_event):
    """
    Function to listen for ESC key press and set the stop event.
    """
    while not stop_event.is_set():
        if keyboard.is_pressed('esc'):
            stop_event.set()
            break
        time.sleep(0.1)  # Sleep to avoid high CPU usage

# Function to capture the keyboard input for direction
def capture_direction():
    while True:
        if keyboard.is_pressed('up'):
            return 'top'
        elif keyboard.is_pressed('down'):
            return 'down'
        elif keyboard.is_pressed('right'):
            return 'right'
        elif keyboard.is_pressed('left'):
            return 'left'

def start_key_listener(stop_event):
    """
    Start the ESC key listener thread.
    """
    key_listener_thread = threading.Thread(target=esc_key_listener, args=(stop_event,))
    key_listener_thread.daemon = True  # Ensure the thread exits with the main program
    key_listener_thread.start()


# Calibration function to determine optimal vibration intensity
def display_intensity(intensity):
    print(f"\nCurrent intensity: {intensity}")

def get_step_value():
    while True:
        try:
            step_value = int(input("Enter the initial increment/decrement step value: "))
            if step_value <= 0:
                print("Step value must be a positive integer. Please try again.")
            else:
                return step_value
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_user_input_for_calibration():
    while True:
        try:
            inputs = []
            print("Enter four numbers for calibration:")
            for i in range(4):
                num = int(input(f"Number {i+1}: "))
                inputs.append(num)
            return inputs
        except ValueError:
            print("Invalid input. Please enter numeric values.")

def select_vibromotor():
    while True:
        print("\nSelect the vibromotor to calibrate:")
        print("0. Finish calibration")
        print("1. Down")
        print("2. Right")
        choice = input("Enter the number corresponding to your choice: ")
        if choice == '1':
            return 60
        elif choice == '2':
            return 120
        elif choice == '0':
            return None  # Finish calibration
        else:
            print("Invalid choice. Please select a valid option.")

def staircase_method(orientation):
    """
    Perform the staircase method to determine the threshold intensity for the given orientation.
    """
    initial_value = 100
    step_sizes = [64, 32, 16, 8, 4, 2, 1, 0]
    step_size_index = 0
    step_size = step_sizes[step_size_index]
    current_value = initial_value
    values = [current_value]
    reversal_points = []
    reversal_indices = []
    direction = None
    direction_changes = 0
    max_reversals = 7
    max_trials = 20
    trial_count = 0

    print("Press 'up' to increase, 'down' to decrease, 'right' to repeat, and 'esc' to exit.")
    
    while direction_changes < max_reversals and trial_count < max_trials:
        key = keyboard.read_event()

        # Only handle the event if it's a key press
        if key.event_type == keyboard.KEY_DOWN:
            if key.name == 'esc':
                break
            elif key.name == 'up':
                if direction == 'down':
                    # Capture reversal index before appending the new value
                    reversal_points.append(current_value)
                    reversal_indices.append(len(values) - 1)
                    direction_changes += 1
                    step_size_index = (step_size_index + 1) % len(step_sizes)
                    step_size = step_sizes[step_size_index]
                    print(f"\nDirection changed to 'up'. New step size: {step_size}")
                direction = 'up'
                current_value += step_size
                if current_value > 100:  # Ensure it doesn't go above 100
                    current_value = 100
                values.append(current_value)
                print(f"Increasing by {step_size}. New value: {current_value}")

            elif key.name == 'down':
                if direction == 'up':
                    # Capture reversal index before appending the new value
                    reversal_points.append(current_value)
                    reversal_indices.append(len(values) - 1)
                    direction_changes += 1
                    step_size_index = (step_size_index + 1) % len(step_sizes)
                    step_size = step_sizes[step_size_index]
                    print(f"\nDirection changed to 'down'. New step size: {step_size}")
                direction = 'down'
                current_value -= step_size
                if current_value < 0:  # Ensure it doesn't go below 0
                    current_value = 0
                values.append(current_value)
                print(f"Decreasing by {step_size}. New value: {current_value}")

            elif key.name == 'right':
                values.append(current_value)
                print(f"\nRepeating value: {current_value}")

            trial_count += 1

            # Send vibration command with current value (intensity)
            if belt_controller:
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.CONTINUOUS,
                    intensity=current_value,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=orientation,
                    pattern_iterations=None,
                    pattern_period=500,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False)
                time.sleep(1)
                belt_controller.stop_vibration()

    # Calculate the average of the last 4 direction changes
    if len(reversal_points) >= 4:
        threshold = sum(reversal_points[-4:]) / 4
        # Only plot the last 4 reversal points
        plot_reversal_points = reversal_points[-4:]
        plot_reversal_indices = reversal_indices[-4:]
    else:
        threshold = sum(reversal_points) / len(reversal_points)
        # Plot all the reversal points if there are less than 4
        plot_reversal_points = reversal_points
        plot_reversal_indices = reversal_indices

    print(f"Estimated Threshold: {threshold:.2f}")

    # Plot the entire sequence of values
    plt.figure()
    plt.plot(values, marker='o', linestyle='-', color='b', label='Values')

    # Plot only the last 4 reversal points with red circles
    if len(reversal_points) > 0:
        plt.plot(plot_reversal_indices, plot_reversal_points, marker='o', linestyle='None', color='r', label='Last 4 Reversal Points')

    plt.xlabel('Trials')
    plt.ylabel('Value')
    plt.title(f'Staircase Method_{orientation}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{participant_ID}_{orientation}_alpha')

    return threshold

def calibrate_intensity(orientation):
    global calibrated_intensity  # Declare that we are using the global variable

    print(f"Calibrating intensity for orientation: {orientation}\n")

    # Run staircase method to determine the optimal intensity
    calibrated_intensity = staircase_method(orientation)
    calibrated_intensity = int(calibrated_intensity)
    print(f'User-calibrated intensity: {calibrated_intensity:.2f}')
    
    if belt_controller:
        belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=calibrated_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False)
        time.sleep(2)
        belt_controller.stop_vibration()

    return calibrated_intensity

# Function to send vibration for a given direction
def vibrate_direction(direction, stop_event, int_top, int_bottom, int_right, int_left):
    # Check if belt_controller is initialized
    if not belt_controller:
        print("Error: Belt controller not initialized.")
        return

    try:
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
        else:
            print(f"Direction '{direction}' not recognized.")
            return

        # Keep vibrating until the stop event is set
        while not stop_event.is_set():
            time.sleep(0.1)

    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        # Ensure vibrations are stopped
        belt_controller.stop_vibration()

# Directions for training
directions = ['top', 'down', 'right', 'left']

# Familiarization phase
def familiarization_phase(int_top, int_bottom, int_right, int_left, avg_int):
    print("\nFamiliarization Phase")
    time.sleep(3)
    
    for direction in directions:        
        while True:
            stop_event = threading.Event()  # Event to stop vibration
            
            print(f"Vibrating for {direction}.")
            # Start the vibration in a separate thread
            vibration_thread = threading.Thread(
                target=vibrate_direction,
                args=(direction, stop_event, int_top, int_bottom, int_right, int_left)
            )
            vibration_thread.start()
            user_response = capture_direction()
            print(f"User response: {user_response}")
            stop_event.set()  # Signal the vibration thread to stop
            vibration_thread.join()  # Wait for the vibration thread to stop
            belt_controller.stop_vibration()
            time.sleep(1)  # Short delay between each trial
            
            if user_response == direction:
                break  # Exit the loop if the user response is correct
            else:
                print("Incorrect response. Please try again.")

    if belt_controller:
        belt_controller.stop_vibration()
        belt_controller.send_pulse_command(
            channel_index=0,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b1111,
            intensity=avg_int,
            on_duration_ms=150,
            pulse_period=500,
            pulse_iterations=5, 
            series_period=5000,
            series_iterations=1,
            timer_option=BeltVibrationTimerOption.RESET_TIMER,
            exclusive_channel=False,
            clear_other_channels=False)

def comfortness():
    while True:
        response = input("\nIs the intensity good for all motors? (yes/no): ").strip().lower()
        if response in ['yes', 'no']:
            return response == 'yes'
        else:
            print("Invalid response. Please enter 'yes' or 'no'.")

def apply_intensity (intensities):
    print("Applying intensities to vibromotors:")
    for orientation, intensity in intensities.items():
        if intensity is not None:
            print(f"Applying intensity {intensity} to orientation {orientation} vibromotor.")

def main_calibration_process():
    """
    Main control loop to orchestrate the steps.
    """

    intensities = {
        90: None,  # Top
        60: None,   # Bottom
        120: None,   # Right
        45: None    # Left
    }
    # applied_intensities = {}
    # Track whether bottom and right calibrations have been performed
    calibrated_bottom = False
    calibrated_right = False

    while True:
        # Calibration and Familiarization with the selected motor
        orientation = select_vibromotor()
        if orientation is None:
            break

        # Calibrate intensity for the selected orientation
        calibrated_intensity = calibrate_intensity(orientation)
        
        if orientation == 60:  # Bottom
            intensities[60] = calibrated_intensity
            intensities[90] = calibrated_intensity  # Top has same intensity as Bottom
            calibrated_bottom = True
        elif orientation == 120:  # Right
            intensities[120] = calibrated_intensity
            intensities[45] = calibrated_intensity  # Left has same intensity as Right
            calibrated_right = True

        print(f"Calibration for orientation {orientation} is complete with intensity {calibrated_intensity}.")

        # Small delay before proceeding to familiarization
        time.sleep(1)
        
        # Proceed with familiarization only if necessary intensities are calibrated
        if calibrated_bottom or calibrated_right:
            # Safely convert to integer
            int_bottom = int(intensities[60] if intensities[60] is not None else 0)
            int_right = int(intensities[120] if intensities[120] is not None else 0)
            
            # Assign corresponding values based on available calibrations
            if calibrated_bottom and calibrated_right:
                int_top = int_bottom
                int_left = int_right
                avg_int = int((int_top + int_right) // 2)
            elif calibrated_bottom:
                int_top = int_bottom
                int_left = int_right = avg_int = int_bottom
            elif calibrated_right:
                int_left = int_right
                int_top = int_bottom = avg_int = int_right

            # Start familiarization phase with the selected intensity
            familiarization_phase(int_top, int_bottom, int_right, int_left, avg_int)


        # Ask if intensity is good
        if comfortness():
            if calibrated_bottom or calibrated_right:
                # At least one of Bottom or Right has been calibrated
                if calibrated_bottom and calibrated_right:
                    # Both Bottom and Right have been calibrated
                    print("\nApplying different intensities based on calibrations for Bottom and Right.")
                    # Apply Bottom intensity to Bottom and Top
                    if intensities[60] is not None:
                        intensities[90] = intensities[60]
                    # Apply Right intensity to Right and Left
                    if intensities[120] is not None:
                        intensities[45] = intensities[120]
                else:
                    # Only one of Bottom or Right has been calibrated
                    print("\nApplying the calibrated intensity to all vibromotors.")
                    if calibrated_bottom:
                        all_intensity = intensities[60]
                    else:
                        all_intensity = intensities[120]
                    # Apply the same intensity to all motors
                    intensities[60] = intensities[90] = intensities[120] = intensities[45] = all_intensity
                    
                int_top = int(intensities[90])
                int_bottom = int(intensities[60])
                int_right = int(intensities[120])
                int_left = int(intensities[45])
                avg_int = int((int_top + int_right) // 2)

                # Apply intensities
                apply_intensity(intensities)
                break
        else:
            print("Repeating calibration and familiarization for the selected motor.")
            continue

    return  int_top, int_bottom, int_right, int_left, avg_int 

def training_task(int_top, int_bottom, int_right, int_left, avg_int):
    directory = r"D:/WWU/M8 - Master Thesis/Project/Code/"
    time.sleep(10)
    print("\nPress 'Enter' to proceed the training task")
    while True:
        if keyboard.is_pressed('enter'):
            break
    print("Training Task will start")
    
    all_set_accuracies = []
    all_block_accuracies = []
    max_sets = 3  # Maximum number of sets allowed
    set_count = 0  # Counter for the number of sets
    all_set_results = {}

    for set_count in range(1, max_sets + 1):
        print(f"Starting Training Set {set_count}")
        correct_responses_per_block = []
        blocks = 3
        trials_per_block = 16
        block_accuracies = []
        block_results = {}
        combined_results = {
            'Trial' : [],
            'Block' : [],
            'Actual Direction' : [],
            'Predicted Direction' : [],
            'Response Time (s)' : []
        }
        for block in range(blocks):
            correct_responses = 0
            time.sleep(5)

            # Create a list with two of each direction and shuffle it
            block_directions = directions * 4
            random.shuffle(block_directions)

            actual_directions =[]
            predicted_directions = []
            response_times = []
            trial_numbers = []

            for trial_num, direction in enumerate(block_directions[:trials_per_block], start = 1):
                print(f"Trial {trial_num}: Vibration direction is {direction}.")
                
                # Setup for vibration
                stop_event = threading.Event()  # Event to stop vibration
                vibration_thread = threading.Thread(
                    target=vibrate_direction,
                    args=(direction, stop_event, int_top, int_bottom, int_right, int_left))
                vibration_thread.start()
                
                start_time = time.time()
                user_response = capture_direction()
                end_time = time.time()
                response_time = end_time - start_time

                print(f"User response: {user_response}")
                stop_event.set()  # Signal the vibration thread to stop
                vibration_thread.join()  # Wait for the vibration thread to stop
                belt_controller.stop_vibration()
                time.sleep(1)

                trial_numbers.append(trial_num)
                actual_directions.append(direction)
                predicted_directions.append(user_response)
                response_times.append(response_time)

                # Add to combined results
                combined_results['Trial'].append(trial_num)
                combined_results['Block'].append(block + 1)
                combined_results['Actual Direction'].append(direction)
                combined_results['Predicted Direction'].append(user_response)
                combined_results['Response Time (s)'].append(response_time)

                if user_response == direction:
                    correct_responses += 1

                # Store the results for the current block
                block_results[f'Block {block + 1}'] = {
                    'Trial': trial_numbers,
                    'Actual Direction': actual_directions, 
                    'Predicted Direction': predicted_directions,
                    'Response Time (s)': response_times
                }   

            # Stop vibration after completing a block with custom stop signal
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
                
            # Calculate accuracy for the block
            block_accuracy = (correct_responses / trials_per_block) * 100
            block_accuracies.append(block_accuracy)
            correct_responses_per_block.append(correct_responses)
            print(f"Block {block + 1} complete. Accuracy: {block_accuracy:.2f}%\n")
            
        # Calculate and store the average accuracy for the set
        set_average_accuracy = np.mean(block_accuracies)
        all_set_accuracies.append(set_average_accuracy)
        all_block_accuracies.append(block_accuracies)
        print(f"Set {set_count} average accuracy: {set_average_accuracy:.2f}%")

        # Save results for the set
        all_set_results[f'Set {set_count}'] = combined_results

        # Determine if the training accuracy is sufficient
        if set_average_accuracy >= 90:
            print(f"Training completed with an accuracy of {set_average_accuracy:.2f}%")
            break
        else: 
            print(f"Training accuracy below 90% with an accuracy of {set_average_accuracy:.2f}%")
            set_count +=1

        print("\nPress 'Enter' to proceed the next training set")
        while True:
            if keyboard.is_pressed('enter'):
                break            
    
    if set_count == max_sets and all(acc < 90 for acc in all_set_accuracies):
        print("Maximum sets reached, but training accuracy is still below 90%.")
              
    # Save result to .txt file
    #directory = r"C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/"
    file_path = f"{directory}training_alpha_{participant_ID}.txt"
    with open(file_path, 'w') as file:  
        file.write(f"Participant ID: {participant_ID}\n")
        file.write(f"Intensity for top and bottom: {int_top}\n")
        file.write(f"Intensity for right and left: {int_left}\n")
        file.write(f"Average intensity: {avg_int}\n")
        file.write(f"Training task: {set_count} set/sets\n")
        for i, accuracies in enumerate(all_block_accuracies, start=1):
            file.write(f"\nSet {i} block accuracies: {accuracies}\n")
            file.write(f"Set {i} average accuracy: {all_set_accuracies[i-1]:.2f}%\n")
        file.write(f"\nOverall training completed with an average accuracy of {np.mean(all_set_accuracies):.2f}%\n")
    print(f'\nResults saved to {file_path}')

    # Excel output
    #with pd.ExcelWriter('C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/training_alpha.xlsx') as writer:
    with pd.ExcelWriter(f'D:/WWU/M8 - Master Thesis/Project/Code/training_alpha_{participant_ID}.xlsx') as writer:
        # Write each set's results to its own sheet
        for set_name, results in all_set_results.items():
            df = pd.DataFrame(results)
            df.to_excel(writer, sheet_name=set_name, index=False)

    return
    #return average_accuracy, block_accuracies, actual_directions, predicted_directions

def visualize_confusion_matrix(excel_file_path, participant_ID):
    # Load the Excel file
    with pd.ExcelFile(excel_file_path) as xls:
        # Iterate over each sheet in the Excel file
        for sheet_name in xls.sheet_names:
            # Load the data from the current sheet
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Extract the actual and predicted directions
            actual_directions = df['Actual Direction']
            predicted_directions = df['Predicted Direction']

            # Compute the confusion matrix
            cm = confusion_matrix(actual_directions, predicted_directions)

            # Plot the confusion matrix using Seaborn
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=df['Actual Direction'].unique(),
                        yticklabels=df['Actual Direction'].unique())
            plt.xlabel('Predicted Direction')
            plt.ylabel('Actual Direction')
            plt.title(f'Confusion Matrix of Actual vs. Predicted Directions_{participant_ID}_{sheet_name}')
            plt.show()
            #plt.savefig(f'D:/WWU/M8 - Master Thesis/Project/Code/confusion_matrix_{participant_ID}_{sheet_name}.jpg')

def save_calibration_data(participant_ID, int_top, int_bottom, int_left, int_right, avg_int):
    directory = r"D:/WWU/M8 - Master Thesis/Project/Code/"
    file_path = os.path.join(directory, f'intensity alpha_{participant_ID}.txt')
    calibration_data = (
        f"int_top: {int_top}\n"
        f"int_bottom: {int_bottom}\n"
        f"int_left: {int_left}\n"
        f"int_right: {int_right}\n"
        f"avg_int: {avg_int}\n")
    
    # Save values to the text file
    with open(file_path, 'w') as file:
        file.write(calibration_data)
    print(f"Calibration data saved to {file_path}")

if __name__ == "__main__":
    participant_ID = input("Enter Participant ID: ")	

    # Calibration 
    int_top, int_bottom, int_right, int_left, avg_int = main_calibration_process()

    # Training task
    training_task(int_top, int_bottom, int_right, int_left, avg_int)

    # Run confusion matrix
    #visualize_confusion_matrix('C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/training_alpha.xlsx')
    visualize_confusion_matrix(f'D:/WWU/M8 - Master Thesis/Project/Code/training_alpha_{participant_ID}.xlsx', participant_ID)
    save_calibration_data(participant_ID, int_top, int_bottom, int_left, int_right, avg_int)

    belt_controller.disconnect_belt() if belt_controller else None
    sys.exit()