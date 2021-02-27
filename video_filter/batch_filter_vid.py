import os
from itertools import product

video_folder_path = "/tank/qizheng/dds-keypoint/video-filter/dataset"
results_folder_path = "/tank/qizheng/vid_comp_master/videos"
video_list = ["surf_4_cropped","surf_5","surf_6","surf_7"]
upper_bound_list = [0.25]
lower_bound_list = [0.0025, 0.005, 0.01]
confidence_threshold = 0.9

for v, ubound, lbound in product(video_list, upper_bound_list, lower_bound_list):
    os.system(
        f"python -W ignore filter_vid.py --video {v} "
        f" --video_folder_path {video_folder_path} "
        f" --result_folder_path {results_folder_path} "
        f" --stats_file_path stats "
        f" --upper_bound {ubound} --lower_bound {lbound} "
        f" --confidence_threshold {confidence_threshold} "
    )
