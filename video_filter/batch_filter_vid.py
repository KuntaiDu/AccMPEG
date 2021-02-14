import os
from itertools import product

video_folder_path = "dataset_original"
results_folder_path = "dataset_filtered"
video_list = ["dance_5"]
#upper_bound_list = [0.15, 0.2, 0.25, 0.3]
upper_bound_list = [0.25, 0.3]
lower_bound_list = [0.01]

for v, ubound, lbound in product(video_list, upper_bound_list, lower_bound_list):
    os.system(
        f"python -W ignore filter_vid.py --video {v} "
        f" --video_folder_path {video_folder_path} "
        f" --result_folder_path {results_folder_path} "
        f" --stats_file_path stats "
        f" --upper_bound {ubound} --lower_bound {lbound} "
    )