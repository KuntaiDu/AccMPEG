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
    "visdrone/videos/vis_%d" % i
    for i in range(169, 174)
    # "large_object/large_%d" % i
    # for i in range(3, 5)
    # "visdrone/videos/vis_172",
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
# model_name = f"COCO_full_normalizedsaliency_R_101_FPN_crossthresh"
model_name = f"COCO_full_normalizedsaliency_R_101_FPN_crossthresh_5xdownsample"

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
conv_list = [5]
bound_list = [0.6]
stats = "stats_gtbbox_FPN"

# app_name = "Segmentation/fcn_resnet50"
app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# app_name = "EfficientDet"
filename = "vgg11_downsample"

for v, conv, bound in product(v_list, conv_list, bound_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    # visdrone/videos/vis_169_blackgen_bound_0.2_qp_30_conv_5_app_FPN.mp4
    # output = f"{v}_blackgen_bound_{bound}_qp_30_conv_{conv}_app_FPN.mp4"
    output = f"{v}_blackgen_downsample_bound_{bound}_conv_{conv}_app_FPN.mp4"

    if True or not os.path.exists(output):
        # if True:
        os.system(
            f"python compress_blackgen.py -i {v}_qp_{base}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output} --tile_size {tile}  -p maskgen_pths/{model_name}.pth.best"
            f" --conv_size {conv} "
            f" -g {v}_qp_{high}.mp4 --bound {bound} --qp {high} --smooth_frames 30 --app {app_name} "
            f"--maskgen_file /tank/kuntai/code/video-compression/maskgen/{filename}.py"
        )
        #     os.system(f"cp {v}_qp_{base}.mp4 {output}.base.mp4")

    # output = "trial/" + output

    # if not os.path.exists(output):
    #     # if True:
    #     os.system(
    #         f"python compress_blackgen.py -i {v}_qp_{base}.mp4 "
    #         f" {v}_qp_{high}.mp4 -s {v} -o {output} --tile_size {tile}  -p maskgen_pths/{model_name}.pth.best"
    #         f" --conv_size {conv} "
    #         f" -g {v}_qp_{high}.mp4 --bound {bound} --qp {high} --smooth_frames 30 --app {app_name}"
    #     )

    os.system(
        f"python inference.py -i {output} --app {app_name} --confidence_threshold 0.3"
    )

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --confidence_threshold 0.3 --gt_confidence_threshold 0.3 --app {app_name} --stats {stats}"
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

