import glob
import os
import subprocess
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]
# v_list = ["youtube_videos/dashcam_%d_crop" % (i + 1) for i in range(4)] + [
#     "youtube_videos/trafficcam_%d_crop" % (i + 1) for i in range(4)
# ]

# v_list = ["youtube_videos/dashcam_%d_crop" % (i + 1) for i in range(4)]
# v_list = ["dashcam/dashcam_2"]
# v_list = ["visdrone/videos/vis_172"]

# v_list = [
#     "dashcam/dashcam_2",
#     # "visdrone/videos/vis_170",
#     # "visdrone/videos/vis_173",
#     # "visdrone/videos/vis_169",
#     # "visdrone/videos/vis_172",
#     # "visdrone/videos/vis_209",
#     # "visdrone/videos/vis_217",
# ]
# v_list = (
#     # ["videos/drone_0"]
#     ["videos/dashcamcropped_%d" % i for i in range(1, 8)]
#     + ["videos/driving_%d" % i for i in range(5)]
# )

v_list = ["videos/dashcamcropped_%d" % i for i in [3, 5]]
# + [
#     "dashcam/dashcam_%d" % i for i in range(1, 11)
# ]
# v_list = ["dashcam/dashcam_%d" % i for i in range(2, 11)]
attr_list = ["FPN"]
# v_list = ["visdrone/videos/vis_171"]
# v_list = [v_list[2]]
# base = 50
# high_list = [30, 34, 38]
tile = 16

# high_low_list = [(30, 42), (34, 46), (38, 50)]
high_low_list = [(30, 36)]
gt = 30
# lower_bound_list = [0.3]


stats = "frozen_stats_MLSys/stats_QP30_thresh7_segmented_FPN"
conf_thresh = 0.7
gt_conf_thresh = 0.7
eaar_conf = 0.7
app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"


visualize_step_size = 300


for v, attr, (high, base) in product(v_list, attr_list, high_low_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_eaar_qp_{base}_{high}_conf_{eaar_conf}_FPN.mp4"

    if not os.path.exists(output):
        # if True:

        subprocess.run(
            [
                "python",
                "compress_eaar.py",
                "-i",
                f"{v}_qp_{base}.mp4",
                f"{v}_qp_{high}.mp4",
                "-s",
                f"{v}",
                "-o",
                f"{output}",
                "--tile_size",
                f"{tile}",
                # "-g",
                # f"{v}_qp_{high}_ground_truth.mp4",
                "--hq",
                f"{high}",
                "--lq",
                f"{base}",
                "--app",
                app_name,
                "--conf",
                f"{eaar_conf}",
            ]
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
