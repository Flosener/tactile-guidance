# region Setup

import time
from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption, BeltVibrationPattern)
from auto_connect import interactive_belt_connect, setup_logger
import threading
import sys
from pynput.keyboard import Key, Listener
import numpy as np
from queue import PriorityQueue
import cv2

# Depth navigation functions

from resources.depth_navigation_functions import map_obstacles, astar, smooth_path, check_obstacles_between_points, find_obstacle_target_point


# endregion

def connect_belt():
    setup_logger()

    belt_controller_delegate = Delegate()
    belt_controller = BeltController(belt_controller_delegate)

    # Interactive script to connect the belt
    interactive_belt_connect(belt_controller)

    if belt_controller.get_connection_state() != BeltConnectionState.CONNECTED:
        print("Connection failed.")
        return False, belt_controller
    else:
        # Change belt mode to APP mode
        belt_controller.set_belt_mode(BeltMode.APP_MODE)
        return True, belt_controller

class Delegate(BeltControllerDelegate):
    # Belt controller delegate
    pass


class BraceletController:
    def __init__(self, vibration_intensities = {'bottom': 50, 'top': 50, 'left': 50, 'right': 50}):
        self.vibration_intensities = vibration_intensities
        self.searching = False
        self.prev_hand = None
        self.prev_target = None
        self.frozen_x, self.frozen_y, self.frozen_w, self.frozen_h = -1, -1, -1, -1
        self.frozen = False
        self.timer = 0
        self.vibrate = True

    def choose_detection(self, bboxes, previous_bbox=None, w=1920, h=1080):
        # Hyperparameters
        track_id_weight = 1000
        exponential_weight = 2
        distance_weight = 100

        candidates = []
        for bbox in bboxes:  # x, y, w, h, id, cls, conf
            # bbox has to be within image dimensions
            if bbox[0] <= w and bbox[1] <= h:
                # confidence score
                confidence = bbox[6]  # in [0,1]
                confidence_score = exponential_weight**confidence - 1  # exponential growth in [0,1], could also use np.exp() and normalize
                # tracking score
                current_track_id = bbox[4]
                previous_track_id = previous_bbox[4] if previous_bbox is not None else -1
                track_id_score = track_id_weight if current_track_id == previous_track_id else 1  # 1|ꝏ
                # distance score
                if previous_bbox is None:
                    distance = None
                    distance_inverted = 1
                else:
                    current_location = bbox[:2]
                    previous_location = previous_bbox[:2]
                    distance = np.linalg.norm(current_location - previous_location)
                    distance_inverted = 1 / distance if distance >= 1 else distance_weight

                # total score
                score = track_id_score * confidence_score * distance_inverted

                # Possible scores:
                # ꝏ -- same trackingID
                # 100 -- different trackingID, matching BBs (max. 1px deviation), conf=1
                # [0,1] -- different trackingID, BBs distance in [1., sqrt(w^2*h^2)], conf=1
                candidates.append(score)
            else:
                candidates.append(0)

        true_detection = bboxes[np.argmax(candidates)] if len(candidates) else previous_bbox

        return true_detection

    def calibrate_intensity(self):
        # to be implemented
        return 50

    def get_bb_bounds(self, BB):
        BB_x, BB_y, BB_w, BB_h = BB[:4]

        BB_right = BB_x + BB_w // 2
        BB_left = BB_x - BB_w // 2
        BB_top = BB_y - BB_h // 2
        BB_bottom = BB_y + BB_h // 2

        return BB_right, BB_left, BB_top, BB_bottom

    def get_intensity(self, handBB, targetBB, vibration_intensities, depth_img):
        # Calculate angle
        xc_hand, yc_hand = handBB[:2]
        xc_target, yc_target = targetBB[:2]
        angle_radians = np.arctan2(yc_hand - yc_target, xc_target - xc_hand)  # inverted y-axis
        angle = np.degrees(angle_radians) % 360

        # Initialize motor intensities
        right_intensity = 0
        left_intensity = 0
        top_intensity = 0
        bottom_intensity = 0

        max_bottom_intensity, max_top_intensity, max_left_intensity, max_right_intensity = vibration_intensities["bottom"], vibration_intensities["top"], vibration_intensities["left"], vibration_intensities["right"]

        # Calculate motor intensities based on the angle
        if 0 <= angle < 90:
            right_intensity = (90 - angle) / 90 * max_right_intensity
            top_intensity = angle / 90 * max_top_intensity
        elif 90 <= angle < 180:
            top_intensity = (180 - angle) / 90 * max_top_intensity
            left_intensity = (angle - 90) / 90 * max_left_intensity
        elif 180 <= angle < 270:
            left_intensity = (270 - angle) / 90 * max_left_intensity
            bottom_intensity = (angle - 180) / 90 * max_bottom_intensity
        elif 270 <= angle < 360:
            bottom_intensity = (360 - angle) / 90 * max_bottom_intensity
            right_intensity = (angle - 270) / 90 * max_right_intensity

        if type(depth_img) != None:
            return int(right_intensity), int(left_intensity), int(top_intensity), int(bottom_intensity), 50

        # front / back motor (depth), currently it is used for grasping signal until front motor is added
        # If there is an anything between hand and target that can be hit (depth smaller than depth of both target and image) - move backwards

        hand_right, hand_left, hand_top, hand_bottom = self.get_bb_bounds(handBB)
        target_right, target_left, target_top, target_bottom = self.get_bb_bounds(targetBB)

        roi_x_min, roi_x_max, roi_y_min, roi_y_max = int(min(hand_right, target_right)), int(max(hand_left, target_left)), int(min(hand_top, target_top)), int(max(hand_bottom, target_bottom))

        if depth_img is not None:
            roi = depth_img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
            try:
                max_depth = np.max(roi)
            except ValueError:
                max_depth = -1

        baseline_depth_intensity = max(vibration_intensities.values())

        if max_depth < handBB[7]:
            depth_intensity = round(-baseline_depth_intensity / 5) * 5

        # Otherwise check if hand is closer or further than the target and set depth intensity accordingly
        else:
            depth_distance = handBB[7] - targetBB[7]
            if isinstance(depth_distance, (int, float, np.integer, np.floating)) and not (np.isnan(depth_distance)):
                if depth_distance > 0:  # move forward
                    depth_intensity = min(int(10000 / depth_distance), baseline_depth_intensity)  # d<=10 -> 100, d=1000 -> 10
                elif depth_distance < 0:  # move backwards
                    depth_intensity = max(int(10000 / depth_distance), -baseline_depth_intensity)  # d<=10 -> -100, d=1000 -> -10
                depth_intensity = round(depth_intensity / 5) * 5  # steps in 5, so users can feel the change (can be replaced by a calibration value later for personalization)
            else:
                depth_intensity = 0  # placeholder

        return int(right_intensity), int(left_intensity), int(top_intensity), int(bottom_intensity), depth_intensity

    def check_overlap(self, handBB, targetBB, frozen=False):
        # Get BB information
        hand_x, hand_y, hand_w, hand_h = handBB[:4]
        target_x, target_y, target_w, target_h = targetBB[:4]

        # Calculate BB bounds to check for overlap
        hand_right = hand_x + hand_w // 2
        hand_left = hand_x - hand_w // 2
        hand_top = hand_y - hand_h // 2
        hand_bottom = hand_y + hand_h // 2

        target_right = target_x + target_w // 2
        target_left = target_x - target_w // 2
        target_top = target_y - target_h // 2
        target_bottom = target_y + target_h // 2

        # all cases of touching any side + handBB inside targetBB
        touched_left = hand_right >= target_left and hand_left <= target_left and hand_top <= target_bottom and hand_bottom >= target_top
        touched_right = hand_left <= target_right and hand_right >= target_right and hand_top <= target_bottom and hand_bottom >= target_top
        touched_top = hand_bottom >= target_top and hand_top <= target_top and hand_right >= target_left and hand_left <= target_right
        touched_bottom = hand_top <= target_bottom and hand_bottom >= target_bottom and hand_right >= target_left and hand_left <= target_right
        is_inside = hand_left >= target_left and hand_right <= target_right and hand_top >= target_top and hand_bottom <= target_bottom
        is_touched = touched_left or touched_right or touched_top or touched_bottom

        # If both BBs touch, keep frozen targetBB size
        if is_touched or is_inside:
            frozen = True
            # only if the center of the hand is in the targetBB send the grasp signal
            if (target_left <= hand_x <= target_right) and (target_top <= hand_y <= target_bottom):
                return True, target_x, target_y, target_w, target_h, frozen
            else:
                return False, target_x, target_y, target_w, target_h, frozen
        # Else, update targetBB size
        else:
            frozen = False
            return False, target_x, target_y, target_w, target_h, frozen

    def navigate_hand(self, belt_controller, bboxes, target_cls: str, hand_clss: list, depth_img, vibration_intensities=None, metric=False):
        """ Function that navigates the hand to the target object. Handles cases when either hand or target is not detected.

        Args:
        - belt_controller -- belt controller object
        - bboxes -- object detections in current frame
        - target_cls -- the target object ID
        - hand_clss -- list of hand IDs
        - depth_img -- depth map of the currently processed frame
        - vibration_intensities -- intensitites of vibrations for each separate bracelet motor (range: 0 - 100)#
        - mode -- guiding algorithm selection, options: grasping, depth_navigation

        Returns:
        - overlapping -- information whether hand and target BBs are overlapping
        - frozen_target -- frozen target BB (updated up until occlusion of hand and target occurs)
        """
        if vibration_intensities is None:
            vibration_intensities = self.vibration_intensities

        overlapping = False

        # Search for object and hand with the highest prediction confidence
        ## Filter for hand detections
        bboxes_hands = [detection for detection in bboxes if detection[5] in hand_clss]
        hand = self.choose_detection(bboxes_hands, self.prev_hand)
        self.prev_hand = hand

        ## Filter for target detections
        bboxes_objects = [detection for detection in bboxes if detection[5] == target_cls]
        target = self.choose_detection(bboxes_objects, self.prev_target)
        self.prev_target = target

        if hand is not None and target is not None:
            # Get varying vibration intensities depending on angle from hand to target
            # Navigation without depth map
            if depth_img is None:
                right_int, left_int, top_int, bot_int, depth_int = self.get_intensity(hand, target, vibration_intensities, depth_img)

            # Navigation with depth map
            else:
                obstacles_mask = map_obstacles(hand, target, depth_img, metric)
                obstacles_between_hand_and_target = check_obstacles_between_points(hand, target, obstacles_mask, 1)

                if not obstacles_between_hand_and_target:
                    right_int, left_int, top_int, bot_int, depth_int = self.get_intensity(hand, target, vibration_intensities, depth_img)
                else:
                    obstacle_target = find_obstacle_target_point(hand, target, obstacles_mask)
                    right_int, left_int, top_int, bot_int, depth_int = self.get_intensity(hand, obstacle_target, vibration_intensities, depth_img)

            # Check whether the hand is overlapping the target and freeze targetBB size if necessary (to avoid BB shrinking on occlusion)
            frozenBB = [self.frozen_x, self.frozen_y, self.frozen_w, self.frozen_h]
            frozen_target = target

            if not self.frozen:
                overlapping, self.frozen_x, self.frozen_y, self.frozen_w, self.frozen_h, self.frozen = self.check_overlap(hand, target, self.frozen)
            elif self.frozen:
                overlapping, self.frozen_x, self.frozen_y, self.frozen_w, self.frozen_h, self.frozen = self.check_overlap(hand, frozenBB, self.frozen)
                frozen_target[:4] = frozenBB

        # 1. Grasping
        if overlapping:
            self.searching = True
            if belt_controller and self.vibrate:
                belt_controller.stop_vibration()
                belt_controller.send_pulse_command(
                    channel_index=1,
                    orientation_type=BeltOrientationType.BINARY_MASK,
                    orientation=0b111100,
                    intensity=abs(depth_int),
                    on_duration_ms=150,
                    pulse_period=300,
                    pulse_iterations=5,
                    series_period=5000,
                    series_iterations=1,
                    timer_option=BeltVibrationTimerOption.RESET_TIMER,
                    exclusive_channel=False,
                    clear_other_channels=False
                )
                self.vibrate = False
                print(f'Previous target: {self.prev_target}')
                self.prev_target = None
                #frozen_target = None
            print("G R A S P !")
            return overlapping, frozen_target

        # 2. Guidance
        if hand is not None and target is not None:
            self.searching = True

            if belt_controller and self.vibrate:
                belt_controller.send_vibration_command(
                    channel_index=0,
                    pattern=BeltVibrationPattern.CONTINUOUS,
                    intensity=right_int,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=120,
                    pattern_iterations=None,
                    pattern_period=100,
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
                    pattern_period=100,
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
                    pattern_period=100,
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
                    pattern_period=100,
                    pattern_start_time=0,
                    exclusive_channel=False,
                    clear_other_channels=False
                )
            return overlapping, frozen_target

        # 3. Target is located and hand can be moved into the frame
        if target is not None:
            self.timer += 1
            if belt_controller and self.vibrate and self.searching:
                self.searching = False
                belt_controller.stop_vibration()
                belt_controller.send_pulse_command(
                    channel_index=0,
                    orientation_type=BeltOrientationType.ANGLE,
                    orientation=60,  # bottom motor
                    intensity=max(vibration_intensities.values()) // 6,
                    on_duration_ms=150,
                    pulse_period=500,
                    pulse_iterations=5,
                    series_period=5000,
                    series_iterations=1,
                    timer_option=BeltVibrationTimerOption.RESET_TIMER,
                    exclusive_channel=False,
                    clear_other_channels=False
                )
            # reset searching flag to send command again
            if self.timer >= 50:
                self.searching = True
                self.timer = 0
            return overlapping, target

        # 4. Target is not in the frame yet.
        else:
            self.timer = 0
            self.searching = True
            if belt_controller and self.vibrate:
                belt_controller.stop_vibration()
            return overlapping, None
