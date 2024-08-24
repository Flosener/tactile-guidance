import matplotlib.pyplot as plt
import keyboard

def staircase_method():
    initial_value = 80
    step_sizes = [20, 10, 8, 6, 4, 2, 1]
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
                    print(f"\n Direction changed to 'up'. New step size: {step_size}")
                direction = 'up'
                current_value += step_size
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
                    print(f"\n Direction changed to 'down'. New step size: {step_size}")
                direction = 'down'
                current_value -= step_size
                values.append(current_value)
                print(f"Decreasing by {step_size}. New value: {current_value}")

            elif key.name == 'right':
                values.append(current_value)
                print(f"\n Repeating value: {current_value}")

            trial_count += 1

    # Include the last value and the last 3 reversal points for threshold calculation
    last_points = reversal_points[-3:] + [values[-1]]
    last_indices = reversal_indices[-3:] + [len(values) - 1]
    
    # Calculate the average of these last 4 points
    threshold = sum(last_points) / 4
    print(f"Estimated Threshold: {threshold:.2f}")

    # Plot the entire sequence of values
    plt.figure()
    plt.plot(values, marker='o', linestyle='-', color='b', label='Values')
    
    # Highlight the last 4 points (including the last value and 3 last reversal points)
    if len(last_points) > 0:
        plt.plot(last_indices, last_points, marker='o', linestyle='None', color='r', label='Last 4 Points')

    plt.xlabel('Trials')
    plt.ylabel('Value')
    plt.title('Staircase Method')
    plt.legend()
    plt.grid(True)
    plt.show()

    return threshold

# Run the staircase method
threshold = staircase_method()
