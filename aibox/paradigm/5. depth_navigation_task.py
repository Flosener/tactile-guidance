"""
This script is using code from the following sources:
- YOLOv5 🚀 by Ultralytics, AGPL-3.0 license, https://github.com/ultralytics/yolov5
- StrongSORT MOT, https://github.com/dyhBUPT/StrongSORT, https://pypi.org/project/strongsort/
- Youtube Tutorial "Simple YOLOv8 Object Detection & Tracking with StrongSORT & ByteTrack" by Nicolai Nielsen, https://www.youtube.com/watch?v=oDALtKbprHg
- https://github.com/zenjieli/Yolov5StrongSORT/blob/master/track.py, original: https://github.com/mikel-brostrom/yolo_tracking/commit/9fec03ddba453959f03ab59bffc36669ae2e932a
"""

# System
import sys
import os
from pathlib import Path

# Use the project file packages instead of the conda packages, i.e. add to system path for import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append(str(parent_dir) + '/yolov5')
sys.path.append(str(parent_dir) + '/strongsort')
sys.path.append(str(parent_dir) + '/midas')
sys.path.append(str(parent_dir) + '/unidepth')

os.chdir(parent_dir)

# Navigation
import controller
from bracelet import BraceletController

# Utility
import keyboard
from playsound import playsound
import threading
import numpy as np
import time

# Output data
import pandas as pd
import json


class DepthNavigationTaskController(controller.TaskController):

    def save_output_data(self):

        df = pd.DataFrame(np.array(self.output_data).reshape(len(self.output_data)//3, 3))

        df.to_csv(self.output_path + f"depth_navigation_task_participant_{self.participant}.csv")


if __name__ == '__main__':

    #check_requirements(requirements='../requirements.txt', exclude=('tensorboard', 'thop'))
    
    weights_obj = 'yolov5s.pt'  # Object model weights path
    weights_hand = 'hand.pt' # Hands model weights path
    weights_tracker = 'osnet_x0_25_market1501.pt' # ReID weights path
  
    source = '1' # image/video path or camera source (0 = webcam, 1 = external, ...)
    mock_navigate = False # Navigate without the bracelet using only print commands
    belt_controller = None
    run_object_tracker = False

    metric = True
    weights_depth_estimator = 'v2-vits14' if metric else 'midas_v21_384' # v2-vits14, v1-cnvnxtl; midas_v21_384, dpt_levit_224

    # EXPERIMENT CONTROLS

    target_objs = ['bottle'] * 6

    participant = 1
    output_path = str(parent_dir) + '/results/'

    try:
        with open(output_path + f"calibration_participant_{participant}.json") as file:
            participant_vibration_intensities = json.load(file)
        print('Calibration intensities loaded succesfully.')
    except:
        while True:
            continue_with_baseline = input('Error while loading the calibration file. Do you want to continue with baseline intensity of 50 for each vibromotor? (y/n)')
            if continue_with_baseline == 'y':
                participant_vibration_intensities = {'bottom': 50,
                                                     'top': 50,
                                                     'left': 50,
                                                     'right': 50,}
                break
            elif continue_with_baseline == 'n':
                sys.exit()

    #

    print(f'\nLOADING CAMERA AND BRACELET')

    # Check camera connection
    try:
        source = str(source)
        print('Camera connection successful')
    except:
        print('Cannot access selected source. Aborting.')
        sys.exit()

    # Check bracelet connection
    if not mock_navigate:
        connection_check, belt_controller = controller.connect_belt()
        if connection_check:
            print('Bracelet connection successful.')
        else:
            print('Error connecting bracelet. Aborting.')
            sys.exit()

    try:
        bracelet_controller = BraceletController(vibration_intensities=participant_vibration_intensities)

        task_controller = DepthNavigationTaskController(weights_obj=weights_obj,  # model_obj path or triton URL # ROOT
                        weights_hand=weights_hand,  # model_obj path or triton URL # ROOT
                        weights_tracker=weights_tracker, # ROOT
                        weights_depth_estimator=weights_depth_estimator,
                        source=source,  # file/dir/URL/glob/screen/0(webcam) # ROOT
                        iou_thres=0.45,  # NMS IOU threshold
                        max_det=1000,  # maximum detections per image
                        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                        view_img=True,  # show results
                        save_txt=False,  # save results to *.txtm)
                        imgsz=(640, 640),  # inference size (height, width)
                        conf_thres=0.7,  # confidence threshold
                        save_conf=False,  # save confidences in --save-txt labels
                        save_crop=False,  # save cropped prediction boxes
                        nosave=True,  # do not save images/videos
                        classes_obj=[1,39,40,41,45,46,47,58,74],  # filter by class /  check coco.yaml file or coco_labels variable in this script
                        classes_hand=[0,1], 
                        #class_hand_nav=[80,81],
                        agnostic_nms=False,  # class-agnostic NMS
                        augment=False,  # augmented inference
                        visualize=False,  # visualize features
                        update=False,  # update all models
                        project='runs/detect',  # save results to project/name # ROOT
                        name='video',  # save results to project/name
                        exist_ok=False,  # existing project/name ok, do not increment
                        line_thickness=3,  # bounding box thickness (pixels)
                        hide_labels=False,  # hide labels
                        hide_conf=False,  # hide confidences
                        half=False,  # use FP16 half-precision inference
                        dnn=False,  # use OpenCV DNN for ONNX inference
                        vid_stride=1,  # video frame-rate stride_obj
                        manual_entry=False, # True means you will control the exp manually versus the standard automatic running
                        run_object_tracker=run_object_tracker,
                        run_depth_estimator=True,
                        mock_navigate=mock_navigate,
                        belt_controller=belt_controller,
                        tracker_max_age=10,
                        tracker_n_init=5,
                        target_objs=target_objs,
                        output_data=[],
                        output_path=output_path,
                        participant=participant,
                        participant_vibration_intensities=participant_vibration_intensities,
                        metric=metric,
                        bracelet_controller=bracelet_controller) # debugging
        
        task_controller.run()

    except KeyboardInterrupt:
        controller.close_app(belt_controller)
    
    # In the end, kill everything
    controller.close_app(belt_controller)