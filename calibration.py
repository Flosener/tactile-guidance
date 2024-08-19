from pybelt.examples_utility import belt_controller_log_to_stdout, interactive_belt_connection, belt_mode_to_string, \
    belt_button_id_to_string

from pybelt.belt_controller import BeltController, BeltConnectionState, BeltControllerDelegate, BeltMode

from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType, BeltVibrationPattern,
                                    BeltVibrationTimerOption)

import time
# import keyboard
from pynput import  keyboard
class Delegate(BeltControllerDelegate):

    def on_belt_mode_changed(self, belt_mode):
        print("Belt mode changed to {}.".format(belt_mode_to_string(belt_mode)))
        print_belt_mode(belt_mode)
    def on_belt_button_pressed(self, button_id, previous_mode, new_mode):
        print("Belt button pressed: {}.".format(belt_button_id_to_string(button_id)))
        print_belt_mode(new_mode)



def calibrate( start, end, belt_controller):
    for i in range(start, end): # hardcoded for the pin out in the controller module
        print("calibrating for motor", i, ": \n Use keyboard UP to increase intesity \n Use keyboard DOWN to decrease intensity \n press F to continue:")
        print(" >> UPDATING INTENSITY FOR MOTOR INDEX: ", i , "\n") 
        with keyboard.Events() as events:
            for event in events:
                if type(event) == keyboard.Events.Press:    
                    print(event)
                    if event.key == keyboard.Key.esc:
                        break
                    elif event.key == keyboard.Key.enter:
                        break
                    elif event.key == keyboard.Key.up:
                        BASE_INTENSITY[i] = clamp(BASE_INTENSITY[i] + 5, 5, 100)
                        print("\t>>MOTOR INDEX: ", i , ": ", BASE_INTENSITY[i], "\n") 
                        belt_controller.send_vibration_command(
                            channel_index=1,
                            pattern=BeltVibrationPattern.CONTINUOUS,
                            intensity=BASE_INTENSITY[i],
                            orientation_type=BeltOrientationType.MOTOR_INDEX,
                            orientation=i,
                            pattern_iterations=None,
                            pattern_period=1000,
                            pattern_start_time=0,
                            exclusive_channel=False,
                            clear_other_channels=False,
                        )
                    # elif event.key == keyboard.Key.down:
                    elif event.key == keyboard.Key.down:
                        BASE_INTENSITY[i] = clamp(BASE_INTENSITY[i] - 5, 5, 100)
                        print("\t>>MOTOR INDEX: ", i , ": ", BASE_INTENSITY[i], "\n") 
                        belt_controller.send_vibration_command(
                            channel_index=1,
                            pattern=BeltVibrationPattern.CONTINUOUS,
                            intensity=BASE_INTENSITY[i],
                            orientation_type=BeltOrientationType.MOTOR_INDEX,
                            orientation=i,
                            pattern_iterations=None,
                            pattern_period=1000,
                            pattern_start_time=0,
                            exclusive_channel=False,
                            clear_other_channels=False,
                        )
                    else:
                        print("Use up or down keys, Press Enter to go to next motor. \n  ")
        print("Calibration complete \n ", "Intensities: ", BASE_INTENSITY, "\n" )
    



#BASE_INTENSITY AT BEGINNING SET TO 20% FOR ALL MOTORS
# using a dictionary, to match with motor index
BASE_INTENSITY = {2: 20, 3: 20, 4: 20, 5: 20}

clamp = lambda n, minn, maxn: max(min(maxn, n), minn) ## Function to restrict range from 0 - 100

def main():

    # Interactive script to connect the belt
    belt_controller_delegate = Delegate()
    belt_controller = BeltController(belt_controller_delegate)
    interactive_belt_connection(belt_controller)
    # belt_controller.set_belt_mode(mode=4)
    if belt_controller.get_connection_state() != BeltConnectionState.CONNECTED:
        print("Connection failed.")
        # return 0
    belt_controller.stop_vibration()
    belt_controller.set_inaccurate_orientation_signal_state(enable_in_compass=False, save_on_belt=True, enable_in_app=True)
    belt_controller.set_belt_mode(BeltMode.APP_MODE)

    
    print("Welcome to the \" FEELSPACE INTENSITY CALIBRATION TOOL\" !")
    print("Please connect the tactile bracelet and press Enter to continue.")

    # Wait for user to continue.
    # while True:
        # if keyboard.pressed("enter"):
        #     break

        # with keyboard.Listener(
        # on_press=on_press) as listener:
        #     listener.join()
    ## Calibrate intensity of Motor 1
    
    calibrate(2, 6, belt_controller)
    exit()

    
main()