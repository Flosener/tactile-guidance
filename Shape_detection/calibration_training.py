import numpy as np
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

#belt_controller = None

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
        print("1. Top")
        print("2. Right")
        print("3. Bottom")
        print("4. Left")
        print("0. Finish calibration")
        choice = input("Enter the number corresponding to your choice: ")
        if choice == '1':
            return 90
        elif choice == '2':
            return 120
        elif choice == '3':
            return 60
        elif choice == '4':
            return 45
        elif choice == '0':
            return None  # Finish calibration
        else:
            print("Invalid choice. Please select a valid option.")

def calibrate_intensity(orientation):
    intensity = int(input("Enter the initial intensity: "))
    print(f"Calibrating will start with initial intensity: {intensity}")

    time.sleep(0.5)
    if belt_controller:
        belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=orientation,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False)
    time.sleep(2)
    belt_controller.stop_vibration()

    increment_step = get_step_value()  # Ask user for the initial step value

    while True:
        display_intensity(intensity)
        print(f"Increment/Decrement step: {increment_step}")
        print("Press 'up' to increment, 'down' to decrement, 'escp' to change step, 'right' to display intensity again, 'esc' to exit.")
        
        while True:
            if keyboard.is_pressed('up'):
                intensity += increment_step
                time.sleep(0.1)  
                print(f"Incremented: {intensity}")
                if belt_controller:
                    belt_controller.send_vibration_command(
                        channel_index=0,
                        pattern=BeltVibrationPattern.CONTINUOUS,
                        intensity=intensity,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=orientation,
                        pattern_iterations=None,
                        pattern_period=500,
                        pattern_start_time=0,
                        exclusive_channel=False,
                        clear_other_channels=False)
                time.sleep(2)
                belt_controller.stop_vibration()
            elif keyboard.is_pressed('down'):
                intensity -= increment_step
                time.sleep(0.1)  
                print(f"Decremented: {intensity}")
                if belt_controller:
                    belt_controller.send_vibration_command(
                        channel_index=0,
                        pattern=BeltVibrationPattern.CONTINUOUS,
                        intensity=intensity,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=orientation,
                        pattern_iterations=None,
                        pattern_period=500,
                        pattern_start_time=0,
                        exclusive_channel=False,
                        clear_other_channels=False)
                time.sleep(2)
                belt_controller.stop_vibration()
            elif keyboard.is_pressed('left'):
                print("Enter pressed. Set new increment/decrement step:")
                increment_step = get_step_value()
                break
            elif keyboard.is_pressed('right'):
                display_intensity(intensity)
                time.sleep(0.1)  # Small delay to avoid multiple prints in one press
                if belt_controller:
                    belt_controller.send_vibration_command(
                        channel_index=0,
                        pattern=BeltVibrationPattern.CONTINUOUS,
                        intensity=intensity,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=orientation,
                        pattern_iterations=None,
                        pattern_period=500,
                        pattern_start_time=0,
                        exclusive_channel=False,
                        clear_other_channels=False)
                time.sleep(2)
                belt_controller.stop_vibration()
            elif keyboard.is_pressed('esc'):  # Using 'esc' to exit the loop
                print("Exiting loop.")
                if belt_controller:
                    belt_controller.stop_vibration()
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.BINARY_MASK,
                        orientation=0b111100,
                        intensity=intensity,
                        on_duration_ms=150,
                        pulse_period=500,
                        pulse_iterations=5, 
                        series_period=5000,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False)
                
                # Get four user inputs and calculate average
                user_inputs = get_user_input_for_calibration()
                intensity = int(sum(user_inputs) / len(user_inputs))
                print(f'User-calibrated intensity: {intensity:.2f}')
                if belt_controller:
                    belt_controller.send_vibration_command(
                        channel_index=0,
                        pattern=BeltVibrationPattern.CONTINUOUS,
                        intensity=intensity,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=orientation,  # down
                        pattern_iterations=None,
                        pattern_period=500,
                        pattern_start_time=0,
                        exclusive_channel=False,
                        clear_other_channels=False)                
                time.sleep(2)
                belt_controller.stop_vibration()
                return intensity

def main_calibration_process():
    global calibrated_intensity
    calibrated_intensity = None  # Initialize as None to indicate it hasn't been set yet

    while True:
        orientation = select_vibromotor()
        if orientation is None:
            print("Finished calibration.")
            break
        else:
            calibrated_intensity = calibrate_intensity(orientation)
            print("Calibration for this vibromotor is complete.")
            time.sleep(1)  # Small delay before asking to calibrate another vibromotor

stop_event = threading.Event()

# Function to send vibration for a given direction
def vibrate_direction(direction, stop_event):
    while not stop_event.is_set():
        if direction == 'top':
            belt_controller.send_vibration_command(
                channel_index=0,
                pattern=BeltVibrationPattern.CONTINUOUS,
                intensity=calibrated_intensity,
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
                intensity=calibrated_intensity,
                orientation_type=BeltOrientationType.ANGLE,
                orientation=60,  # down
                pattern_iterations=None,
                pattern_period=500,
                pattern_start_time=0,
                exclusive_channel=False,
                clear_other_channels=False)
        elif direction == 'right':
            belt_controller.send_vibration_command(
                channel_index=0,
                pattern=BeltVibrationPattern.CONTINUOUS,
                intensity=calibrated_intensity,
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
                intensity=calibrated_intensity,
                orientation_type=BeltOrientationType.ANGLE,
                orientation=45,  # left
                pattern_iterations=None,
                pattern_period=500,
                pattern_start_time=0,
                exclusive_channel=False,
                clear_other_channels=False)
        elif direction == 'top right':
            # Alternate between top and right until stop_event is set
            while not stop_event.is_set():
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.SINGLE_SHORT,
                    intensity=calibrated_intensity,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=90,  # Top
                    pattern_iterations=None,
                    pattern_period=500,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False)
                #time.sleep(0.25)  # Delay between vibrations
                if stop_event.is_set():
                    break
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.SINGLE_SHORT,
                    intensity=calibrated_intensity,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=120,  # Right
                    pattern_iterations=None,
                    pattern_period=500,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False)
                #time.sleep(0.25)  # Delay between vibrations
                if stop_event.is_set():
                    break
        elif direction == 'top left':
            # Alternate between top and right until stop_event is set
            while not stop_event.is_set():
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.SINGLE_SHORT,
                    intensity=calibrated_intensity,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=90,  # Top
                    pattern_iterations=None,
                    pattern_period=500,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False)
                #time.sleep(0.5)  # Delay between vibrations
                if stop_event.is_set():
                    break
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.SINGLE_SHORT,
                    intensity=calibrated_intensity,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=45,  # Right
                    pattern_iterations=None,
                    pattern_period=500,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False)
                if stop_event.is_set():
                    break
                #time.sleep(0.5)  # Delay between vibrations
        elif direction == 'bottom right':
            # Alternate between top and right until stop_event is set
            while not stop_event.is_set():
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.SINGLE_SHORT,
                    intensity=calibrated_intensity,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=60,  # down
                    pattern_iterations=None,
                    pattern_period=500,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False)
                #time.sleep(0.5)  # Delay between vibrations
                if stop_event.is_set():
                    break
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.SINGLE_SHORT,
                    intensity=calibrated_intensity,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=120,  # Right
                    pattern_iterations=None,
                    pattern_period=500,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False)
                if stop_event.is_set():
                    break
                #time.sleep(0.5)  # Delay between vibrations
        elif direction == 'bottom left':
            # Alternate between top and right until stop_event is set
            while not stop_event.is_set():
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.SINGLE_SHORT,
                    intensity=calibrated_intensity,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=60,  # down
                    pattern_iterations=None,
                    pattern_period=500,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False)
                #time.sleep(0.5)  # Delay between vibrations
                if stop_event.is_set():
                    break
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.SINGLE_SHORT,
                    intensity=calibrated_intensity,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=45,  # left
                    pattern_iterations=None,
                    pattern_period=500,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False)
                #time.sleep(0.5)  # Delay between vibrations
                if stop_event.is_set():
                    break
        else:
            print(f"Direction '{direction}' not recognized.")
            break

    belt_controller.stop_vibration()


# Calibrate the bracelet intensity
calibrated_intensity = calibrate_intensity()
print(f'Calibrated intensity: {calibrated_intensity}')
#calibrated_intensity = 80

# Directions for training
directions = ['top', 'down', 'right', 'left', 'top right', 'bottom right', 'top left', 'bottom left']

# Function to capture the keyboard input for direction
def capture_direction():
    while True:
        if keyboard.is_pressed('up'):
            time.sleep(0.1)  # Debounce delay
            if keyboard.is_pressed('right'):
                return 'top right'
            elif keyboard.is_pressed('left'):
                return 'top left'
            return 'top'
        elif keyboard.is_pressed('down'):
            time.sleep(0.1)  # Debounce delay
            if keyboard.is_pressed('right'):
                return 'bottom right'
            elif keyboard.is_pressed('left'):
                return 'bottom left'
            return 'down'
        elif keyboard.is_pressed('right'):
            time.sleep(0.1)  # Debounce delay
            if keyboard.is_pressed('up'):
                return 'top right'
            elif keyboard.is_pressed('down'):
                return 'bottom right'
            return 'right'
        elif keyboard.is_pressed('left'):
            time.sleep(0.1)  # Debounce delay
            if keyboard.is_pressed('up'):
                return 'top left'
            elif keyboard.is_pressed('down'):
                return 'bottom left'
            return 'left'

# Familiarization phase
def familiarization_phase():
    print("\nFamiliarization Phase")
    time.sleep(10)
    
    for direction in directions:
        stop_event = threading.Event()  # Event to stop vibration
        
        # Start the vibration in a separate thread
        vibration_thread = threading.Thread(target=vibrate_direction, args=(direction, stop_event))
        vibration_thread.start()
        
        while True:
            print(f"Vibrating for {direction}.")
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
            orientation=0b111100,
            intensity=calibrated_intensity,
            on_duration_ms=150,
            pulse_period=500,
            pulse_iterations=5, 
            series_period=5000,
            series_iterations=1,
            timer_option=BeltVibrationTimerOption.RESET_TIMER,
            exclusive_channel=False,
            clear_other_channels=False)

def training_task():
    time.sleep(10)
    print("\nTraining Task will start")
    
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
        block_directions = directions * 2
        random.shuffle(block_directions)

        actual_directions =[]
        predicted_directions = []
        response_times = []
        trial_numbers = []

        for trial_num, direction in enumerate(block_directions[:trials_per_block], start = 1):
            print(f"Trial {trial_num}: Vibration direction is {direction}.")
            
            # Setup for vibration
            stop_event = threading.Event()  # Event to stop vibration
            vibration_thread = threading.Thread(target=vibrate_direction, args=(direction, stop_event))
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
                intensity=calibrated_intensity,
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
        
    # Calculate and display the average accuracy across all blocks
    average_accuracy = np.mean(block_accuracies)
    print(f"Selected intensity after training: {calibrated_intensity}")
    print(f"Block accuracy: {block_accuracies}")

    # Determine if the training accuracy is sufficient
    if average_accuracy >= 90:
        print(f"Training completed with an accuracy of {average_accuracy:.2f}%")
    else: 
        print(f"Training accuracy below 90% with an accuracy of {average_accuracy:.2f}%")


    # Save result to .txt file
    #file_path = r"C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/training_result.txt"
    file_path = r"D:/WWU/M8 - Master Thesis/Project/Code/training_result.txt"
    with open(file_path, 'w') as file:  
        file.write(f"Selected intensity after training: {calibrated_intensity}\n")
        file.write(f"Block accuracy: {block_accuracies}\n")
        file.write(f"Training completed with an average accuracy of {average_accuracy:.2f}%\n")
    print('\nResults saved to training_result.txt')

    # Excel output
    #with pd.ExcelWriter('C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/training_result.xlsx') as writer:
    with pd.ExcelWriter('D:/WWU/M8 - Master Thesis/Project/Code/training_result.xlsx') as writer:
        # Write the combined results to the first sheet
        combined_df = pd.DataFrame(combined_results)
        combined_df.to_excel(writer, sheet_name='All Blocks', index=False)
        
        # Write each block's results to subsequent sheets
        for block_name, data in block_results.items():
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=block_name, index=False)
    
    #result = {'Actual Direction': actual_directions,
    #     'Predicted Direction': predicted_directions,
    #     'Response Time (s)': response_times}
    #df = pd.DataFrame(data=result)
    #df.to_excel("C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/training_result.xlsx")

    return
    #return average_accuracy, block_accuracies, actual_directions, predicted_directions

def visualize_confusion_matrix(excel_file_path):
    # Load the first sheet from the Excel file
    df = pd.read_excel(excel_file_path, sheet_name='All Blocks')

    #print(df)

    # Extract the actual and predicted directions
    actual_directions = df['Actual Direction']
    predicted_directions = df['Predicted Direction']

    # Compute the confusion matrix
    cm = confusion_matrix(actual_directions, predicted_directions)

    #print(cm)

    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=df['Actual Direction'].unique(),
                yticklabels=df['Actual Direction'].unique())
    plt.xlabel('Predicted Direction')
    plt.ylabel('Actual Direction')
    plt.title('Confusion Matrix of Actual vs. Predicted Directions')
    plt.show()
    plt.savefig('D:/WWU/M8 - Master Thesis/Project/Code/confusion_matrix.jpg')

if __name__ == "__main__":
    main_calibration_process()
    
    # Run familiarization phase
    familiarization_phase()

    # Run training task
    training_task()

    # Run confusion matrix
    #visualize_confusion_matrix('C:/Users/feelspace/OptiVisT/tactile-guidance/Shape_detection/training_result.xlsx')
    visualize_confusion_matrix('D:/WWU/M8 - Master Thesis/Project/Code/training_result.xlsx')

    belt_controller.disconnect_belt() if belt_controller else None
    sys.exit()