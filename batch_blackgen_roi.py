import os
import subprocess
from itertools import product
from config import settings

import yaml

x264_dir = settings.x264_dir

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]
# v_list = ["youtube_videos/dashcam_%d_crop" % (i + 1) for i in range(4)] + [
#     "youtube_videos/trafficcam_%d_crop" % (i + 1) for i in range(4)
# ]

# v_list = ["dashcam/dashcam_%d" % i for i in [2, 5, 6, 8]]
# v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)] + [
#     "dashcam/dashcam_%d" % i for i in range(1, 11)
# ]
# v_list = ["adapt/drive_%d" % i for i in range(30, 60)]
# v_list = ["dashcam/dashcam_%d" % i for i in [7]]


# v_list = v_list[::-1]
# v_list = [v_list[1]]
# v_list = ["dashcam/dashcam_2"]
# v_list = [v_list[2]]
# v_list = ["visdrone/videos/vis_171"]

high = 30
tile = 16
# model_name = f"COCO_full_normalizedsaliency_R_101_FPN_crossthresh"


"""
    For object detection, use bound 0.5, conv 9 for drone videos and dashcam videos.
    Use
    COCO_full_normalizedsaliency_R_101_FPN_crossthresh
    as the model, and use
    ["dashcam/dashcam_%d" % i for i in range(1, 8)]
    and
    ["visdrone/videos/vis_%d" % i for i in range(169, 174)]
    for video id
"""
# conv_list = [3]
# bound_list = [0.05]

#
# for visdrone
# conv_list = [11]
# bound_list = [0.1]
# uniform color background

# conv_list = [1, 5, 9]
# bound_list = [0.15, 0.2, 0.25]
# base_list = [40, 36]

# conv_list = [1, 5]
# bound_list = [0.15, 0.1]
# base_list = [36]

# conv_list = [1]
# bound_list = [0.02]
# base_list = [51]


# conv_list = [1]
# bound_list = [0.1, 0.2]
# base_list = [40]

conv_list = [1]
bound_list = [0.2]
base_list = [40]

# conv_list = [1]
# bound_list = [0.2]
# base_list = [40]

# conv_list = [1, 5]
# bound_list = [0.1, 0.15, 0.05]
# base_list = [-1]
# model_name = f"cityscape_detection_FPN_SSD_withconfidence_allclasses_new_unfreezebackbone"

# v_list = ["videos/drone_%d" % i for i in range(7)] + [
#     "videos/dashcamcropped_%d" % i for i in range(1, 11)
# ]
# v_list = ["videos/drone_%d" % i for i in range(7)]
# v_list = [v_list[i] for i in range(len(v_list)) if i % 2 == 0]
# v_list = ["videos/driving_%d" % i for i in range(4, 5)]
# v_list = ["videos/dashcamcropped_%d" % i for i in range(1, 8)] + [
#     "videos/driving_%d" % i for i in range(5)
# ]
v_list = ["videos/dashcamcropped_%d" % i for i in range(1, 2)]
# v_list = ["videos/surf_%d_final" % i for i in [1, 2, 3, 4, 6, 7]]

# v_list = ["videos/dashcamcropped_%d" % i for i in range(1, 2)]


# FPN
stats = "artifact/stats_QP30_thresh7_segmented_FPN"
conf_thresh = 0.7
gt_conf_thresh = 0.7
app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

# efficientdet
# stats = "frozen_stats_MLSys/stats_QP30_thresh4_segment_EfficientDet"
# conf_thresh = 0.4
# gt_conf_thresh = 0.4
# app_name = "EfficientDet"

# YoLo
# stats = "frozen_stats_MLSys/stats_QP30_thresh3_segment_Yolo"
# stats = "artifact/stats_QP30_thresh3_segment_Yolo"
# conf_thresh = 0.3
# gt_conf_thresh = 0.3
# app_name = "Yolo5s"

# segmentation
# stats = "frozen_stats_MLSys/stats_QP30_segment_fcn"
# app_name = "Segmentation/fcn_resnet50"
# conf_thresh = 0.7
# gt_conf_thresh = 0.7

model_app = "FPN"
model_name = f"COCO_detection_{model_app}_SSD_withconfidence_allclasses_new_unfreezebackbone_withoutclasscheck"
# model_name = "pretrainedkeypointmodel"
# model_app = "fcn"


visualize_step_size = 10000
# accs = [filter([fmt % i, "newSSDwconf", "bound_0.2", "lq_40", "conv_1"]) for i in ids]

import glob

# app_name = "Segmentation/fcn_resnet50"
# app_name = "EfficientDet"
filename = "SSD/accmpegmodel"

for conv, bound, base, v in product(conv_list, bound_list, base_list, v_list):

    print(v, conv, bound, base)

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    # visdrone/videos/vis_169_blackgen_bound_0.2_qp_30_conv_5_app_FPN.mp4
    # output = f"{v}_blackgen_bound_{bound}_qp_30_conv_{conv}_app_FPN.mp4"

    output = f"{v}_roi_bound_{bound}_conv_{conv}_hq_{high}_lq_{base}_app_{model_app}.mp4"

    # examine_output = (
    #     f"{v}_blackgen_dual_SSD_bound_{bound}_conv_{conv}_app_FPN.mp4"
    # )

    # os.system(f"rm -r {examine_output}*")

    if not os.path.exists(output):
    # if True:

        os.system(
            f"python compress_blackgen_roi.py -i {v}_qp_{high}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output} --tile_size {tile}  -p maskgen_pths/{model_name}.pth.best"
            f" --conv_size {conv} "
            f" -g {v}_qp_{high}.mp4 --bound {bound} --hq {high} --lq {base} --smooth_frames 10 --app {app_name} "
            f"--maskgen_file {x264_dir}/../video-compression/maskgen/{filename}.py --visualize_step_size {visualize_step_size}"
        )

    os.system(
        f"python inference.py -i {output} --app {app_name} --confidence_threshold {conf_thresh} --gt_confidence_threshold {gt_conf_thresh} --visualize_step_size {visualize_step_size} "
        # f" --visualize --lq_result {v}_qp_{base}.mp4 --ground_truth {v}_qp_{high}.mp4"
    )

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --confidence_threshold {conf_thresh}  --gt_confidence_threshold {gt_conf_thresh} --app {app_name} --stats {stats}"
    )

    # if not os.path.exists(f"diff/{output}.gtdiff.mp4"):
    #     gt_output = f"{v}_compressed_blackgen_gt_bbox_conv_{conv}.mp4"
    #     subprocess.run(
    #         [
    #             "python",
    #             "diff.py",
    #             "-i",
    #             output,
    #             gt_output,
    #             "-o",
    #             f"diff/{output}.gtdiff.mp4",
    #         ]
    #     )
