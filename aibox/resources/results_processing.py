import os
from pathlib import Path
import pandas as pd

import ast

file = Path(__file__).resolve()
root = file.parents[0]
os.chdir(root)

participant = 1
task = 'm' # g - grasping, m - multiple_objects, d - depth_navigation
column_names = ['Target_class', 'Start_time', 'Navigation_time', 'Freezing_time', 'Grasping_time', 'End_time', 'Button_pressed', 'Detection', 'Confidence', 'Potential_targets', 'Selected_target', 'Target_position']

def load_results_file(task_code, participant):

    task_map = {'g': 'grasping',
                'm': 'multiple_objects',
                'd': 'depth_navigation'}

    task = task_map[task_code]

    df = pd.read_csv(f'../results/{task}/{task}_participant_{participant}.csv')

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    return df


def add_header(df, header):

    df.columns = header


def process_detections(df, detections_column_name):

    df['Detection_hit_count'] = df[detections_column_name].apply(lambda x: ast.literal_eval(x).count(1))
    df['Detection_miss_count'] = df[detections_column_name].apply(lambda x: ast.literal_eval(x).count(0))

    print(df)


def process_confidence(df, confidence_column_name):

    df['Confidence_miss_count'] = df[confidence_column_name].apply(lambda x: ast.literal_eval(x).count(0))
    df['Confidence_hit_count'] = df[confidence_column_name].apply(lambda x: len(ast.literal_eval(x)))

    print(df)


df = load_results_file(task, participant)
add_header(df, column_names)
process_detections(df, 'Detection')
process_confidence(df, 'Confidence')