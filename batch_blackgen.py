import os
import subprocess
from itertools import product

import yaml

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


v_list = [
    # "visdrone/videos/vis_%d" % i
    # for i in range(169, 174)
    # "dashcam/dashcam_2",
    # "large_object/large_%d" % i
    # for i in range(3, 5)
    # "visdrone/videos/vis_172",
    # "dashcam/dashcam_2_short",
    # "dashcam/dashcam_8"
    # "videos/trafficcam/trafficcam_1"
    "dashcam/dashcamcropped_%d" % i
    for i in range(1, 11)
    # "dashcam/dashcamcropped_%d" % i
    # for i in [1, 2, 3, 4, 6, 7]
    # "dashcam/dashcam_2"
    # "visdrone/videos/vis_170",
    # "visdrone/videos/vis_173",
    # "visdrone/videos/vis_169",
    # "visdrone/videos/vis_172",
    # "visdrone/videos/vis_209",
    # "visdrone/videos/vis_217",
]  # + ["dashcam/dashcam_%d" % i for i in range(1, 11)]
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
conv_list = [1, 5]
bound_list = [0.1, 0.15, 0.05]
base_list = [-1]
model_name = f"cityscape_detection_FPN_SSD_withconfidence_allclasses_new_unfreezebackbone"

stats = "stats_FPN_QP30_thresh75_dashcamcropped"
conf_thresh = 0.8
gt_conf_thresh = 0.75
visualize_step_size = 200

# accs = [filter([fmt % i, "newSSDwconf", "bound_0.2", "lq_40", "conv_1"]) for i in ids]

import glob

# app_name = "Segmentation/fcn_resnet50"
app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# app_name = "EfficientDet"
filename = "SSD/accmpegmodel"

for conv, bound, base, v in product(conv_list, bound_list, base_list, v_list):

    print(v, conv, bound, base)

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    # visdrone/videos/vis_169_blackgen_bound_0.2_qp_30_conv_5_app_FPN.mp4
    # output = f"{v}_blackgen_bound_{bound}_qp_30_conv_{conv}_app_FPN.mp4"

    output = f"{v}_blackgen_weight2_thresh75_newSSDwconf_bound_{bound}_conv_{conv}_hq_{high}_lq_{base}_protect_app_FPN.mp4"

    # examine_output = (
    #     f"{v}_blackgen_dual_SSD_bound_{bound}_conv_{conv}_app_FPN.mp4"
    # )

    # os.system(f"rm -r {examine_output}*")

    if True:

        os.system(f"rm -r {output}*")

        os.system(
            f"python compress_blackgen.py -i {v}_qp_{high}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output} --tile_size {tile}  -p maskgen_pths/{model_name}.pth.best"
            f" --conv_size {conv} "
            f" -g {v}_qp_{high}.mp4 --bound {bound} --hq {high} --lq {base} --smooth_frames 30 --app {app_name} "
            f"--maskgen_file /tank/kuntai/code/video-compression/maskgen/{filename}.py --visualize_step_size {visualize_step_size}"
        )

        os.system(
            f"python inference.py -i {output} --app {app_name} --confidence_threshold {conf_thresh} --gt_confidence_threshold {gt_conf_thresh} -g {v}_qp_{high}.mp4 --visualize_step_size {visualize_step_size}"
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

