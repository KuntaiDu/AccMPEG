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
    # "large_object/large_%d" % i
    # for i in range(3, 5)
    # "visdrone/videos/vis_169",
    'visdrone/videos/vis_%d' % i for i in [170, 172, 173]
    # "visdrone/videos/vis_171",
    # "visdrone/videos/vis_170",
    # "visdrone/videos/vis_173",
    # "visdrone/videos/vis_169",
    # "visdrone/videos/vis_172",
    # "visdrone/videos/vis_209",
    # "visdrone/videos/vis_217",
]  # + ["dashcam/dashcam_%d" % i for i in range(1, 11)]
# v_list = [v_list[2]]
# v_list = ["visdrone/videos/vis_171"]
base = 50
high = 30
tile = 16
model_name = f"COCO_full_normalizedsaliency_R_101_FPN_crossthresh"
conv_list = [9]
bound_list = [0.5]
stats = "stats_FPN"

# app_name = "Segmentation/fcn_resnet50"
app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

for v in v_list:

    os.system(
        f"python inference.py -i {v}_autoencoder.mp4 --app {app_name} --confidence_threshold 0.7"
    )

    os.system(
        f"python examine.py -i {v}_autoencoder.mp4 -g {v}_qp_{high}.mp4 --confidence_threshold 0.7 --gt_confidence_threshold 0.7 --app {app_name} --stats {stats}"
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

