import json
import logging
import os
import subprocess
from itertools import product
from pathlib import Path

import coloredlogs
from munch import Munch

from utilities.compressor import h264_compressor_segment

# for name, attr in product(names, attrs):

#     app = f"COCO-Detection/faster_rcnn_R_101_{attr}_3x.yaml"

#     subprocess.run(
#         [
#             "python",
#             "examine_reducto.py",
#             "--input",
#             f"visdrone/videos/{name}_reducto_qp_28_{attr}.mp4",
#             "--source",
#             f"visdrone/videos/{name}",
#             "-g",
#             f"visdrone/videos/{name}_qp_30.mp4",
#             "--stats",
#             f"stats_{attr}",
#             "--json",
#             f"baselines/jsons/{name}.json",
#             "--app",
#             app,
#         ]
#     )

# vnames = ["videos/dashcamcropped_%d" % i for i in range(1, 8)] + [
#     "videos/driving_%d" % i for i in range(5)
# ]
# jsons = ["baselines/jsons/dashcam_%d.json" % i for i in range(1, 8)] + [
#     "baselines/jsons/driving_%d.json" % i for i in range(5)
# ]

vnames = ["videos/dashcamcropped_2"]
jsons = ["baselines/jsons/dashcam_2"]

qp = 31
high = 30
visualize_step_size = 10000
# attrs = ["C4", "FPN", "DC5"]

coloredlogs.install(
    fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
    level="INFO",
)
logger = logging.getLogger("reducto")


# FPN
# stats = "frozen_stats_MLSys/stats_QP30_thresh7_segmented_FPN"
# conf_thresh = 0.7
# gt_conf_thresh = 0.7
# app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

# efficientdet
stats = "frozen_stats_MLSys/stats_QP30_thresh4_segment_EfficientDet"
conf_thresh = 0.4
gt_conf_thresh = 0.4
app_name = "EfficientDet"

# YoLo
# stats = "frozen_stats_MLSys/stats_QP30_thresh3_segment_Yolo"
# conf_thresh = 0.3
# gt_conf_thresh = 0.3
# app_name = "Yolo5s"


# for name, attr in product(names, attrs):
for vname, json_name in zip(vnames, jsons):

    output_name = f"{vname}_reducto_qp_{qp}.mp4"
    output = output_name

    if not os.path.exists(output_name):

        os.system(f"mkdir {vname}_reducto")

        nimages = len(list(Path(vname).iterdir()))

        with open(json_name, "r") as f:
            reducto_fids = json.load(f)

        for dst_fid in range(nimages):

            src_fid = max([i for i in reducto_fids if i <= dst_fid])
            if dst_fid % 100 == 0:
                print(dst_fid, src_fid)

            subprocess.run(
                [
                    "cp",
                    f"{vname}/%010d.png" % src_fid,
                    f"{vname}_reducto/%010d.png" % dst_fid,
                ]
            )

        new_args = Munch()

        new_args.source = f"{vname}_reducto"
        new_args.qp = qp
        new_args.smooth_frames = 10

        h264_compressor_segment(new_args, logger)

    os.system(
        f"python inference.py -i {output} --app {app_name} --confidence_threshold {conf_thresh} --gt_confidence_threshold {gt_conf_thresh} --visualize_step_size {visualize_step_size} "
        # f" --visualize --lq_result {v}_qp_{base}.mp4 --ground_truth {v}_qp_{high}.mp4"
    )

    os.system(
        f"python examine.py -i {output} -g {vname}_qp_{high}.mp4 --confidence_threshold {conf_thresh}  --gt_confidence_threshold {gt_conf_thresh} --app {app_name} --stats {stats}"
    )

