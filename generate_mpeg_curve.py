import argparse
import logging
import os
import subprocess
from pathlib import Path

import coloredlogs
from munch import Munch

from utilities.compressor import h264_compressor_segment
from utilities.results_utils import read_results

attr = "mp4"

# qp_list = [32]

# gt_qp = 20
# qp_list = [20, 21, 22, 24, 26, 30, 34, 40]

# gt_qp = 50
# qp_list = [50, 51, 52, 53, 54, 56, 58, 60, 62]
# qp_list = [32, 42]
# quality_list = [
#     "veryfast",
#     "faster",
#     "fast",
#     "medium",
#     "slow",
#     "slower",
#     "veryslow",
# ]

# gt_qp = 30
# qp_list = [30, 50]
# attr = "mp4"


def main(args):

    gt_qp = args.gt_qp

    logger = logging.getLogger("mpeg_curve")

    for video_name in args.inputs:
        assert Path(video_name).is_dir()
        video_name = Path(video_name)

        # # generate ground truth
        # output_names = []
        # for quality in quality_list:
        #     input_name = f"{video_name}/%010d.png"
        #     output_name = f"{video_name}_qp_{gt_qp}_{quality}.hevc"
        #     output_names.append(output_name)
        #     print(f"Generate video for {output_name}")
        #     # encode_with_qp(input_name, output_name, qp, args)

        #     if args.force or not os.path.exists(output_name):
        #         # encode
        #         subprocess.run(
        #             [
        #                 "ffmpeg",
        #                 "-y",
        #                 "-i",
        #                 input_name,
        #                 "-start_number",
        #                 "0",
        #                 "-qp",
        #                 f"{gt_qp}",
        #                 "-preset",
        #                 f"{quality}",
        #                 output_name,
        #             ]
        #         )
        #         # and inference
        #         subprocess.run(["python", "inference.py", "-i", output_name])

        # subprocess.run(
        #     ["python", "merge_ground_truth.py", "-i"]
        #     + output_names
        #     + ["-o", f"{video_name}_qp_{gt_qp}_ground_truth.hevc"]
        # )

        # generate mpeg curve
        for qp in args.qp_list:
            input_name = f"{video_name}/%010d.png"
            output_name = f"{video_name}_qp_{qp}.{attr}"
            print(f"Generate video for {output_name}")
            # encode_with_qp(input_name, output_name, qp, args)

            if args.force or not os.path.exists(output_name):
                # if True:

                if attr == "hevc":

                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            input_name,
                            "-start_number",
                            "0",
                            "-c:v",
                            "libx265",
                            "-x265-params",
                            f"qp={qp}",
                            output_name,
                        ]
                    )
                elif attr == "mp4":

                    new_args = Munch()
                    new_args.source = str(video_name)
                    new_args.qp = qp
                    new_args.smooth_frames = args.smooth_frames

                    h264_compressor_segment(new_args, logger)

                elif attr == "webm":

                    print("here")

                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            input_name,
                            "-start_number",
                            "0",
                            "-c:v",
                            "libvpx-vp9",
                            "-crf",
                            f"{qp}",
                            "-b:v",
                            "0",
                            "-threads",
                            "8",
                            output_name,
                        ]
                    )

            subprocess.run(
                [
                    "python",
                    "inference.py",
                    "-i",
                    output_name,
                    "--app",
                    args.app,
                    "--visualize_step_size",
                    "1000"
                    # "--confidence_threshold",
                    # "0.95",
                ]
            )

            subprocess.run(
                [
                    "python",
                    "examine.py",
                    "-i",
                    output_name,
                    "-g",
                    f"{video_name}_qp_{gt_qp}.{attr}",
                    "--app",
                    args.app,
                    "--confidence_threshold",
                    f"{args.confidence_threshold}",
                    "--gt_confidence_threshold",
                    f"{args.gt_confidence_threshold}",
                    "--stats",
                    args.stats,
                ]
            )


if __name__ == "__main__":

    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )

    args = Munch()
    # args.inputs = [
    #     "visdrone/videos/vis_%d" % i for i in [169, 170, 171, 172, 173, 209, 217]
    # ]
    # args.inputs = ["dashcam/dashcam_%d" % (i + 1) for i in [9]]
    # args.inputs = [
    #     # "visdrone/videos/vis_171",
    #     # "visdrone/videos/vis_170",
    #     # "visdrone/videos/vis_173",
    #     # "visdrone/videos/vis_169",
    #     # "visdrone/videos/vis_172",
    # ]
    # args.inputs = ["DAVIS/videos/DAVIS_1"]
    # args.inputs = ["visdrone/videos/vis_%d" % i for i in range(169, 174)] + [
    #     "dashcam/dashcam_%d" % i for i in range(1, 11)
    # ]
    # args.inputs = ["dashcam/dashcam_%d" % i for i in range(5, 11)]
    # args.inputs = ["adapt/drive_%d" % i for i in range(60)]
    # args.inputs = ["visdrone/videos/vis_%d" % i for i in [169]]
    # args.inputs = ["large_object/large_%d" % i for i in range(1, 5)]
    # args.inputs = ["dashcam/dashcam_%d" % i for i in range(4, 11)]
    # args.inputs = ["dashcam/dashcam_%d" % i for i in range(1, 8)]
    # args.inputs = ["dashcam/dashcam_%d" % i for i in [2, 5, 6, 8]]
    # args.inputs = ["visdrone/videos/vis_171"]
    # args.gt_qp = 20
    # args.qp_list = [20, 21, 22, 24, 26, 30, 34, 40]
    args.gt_qp = 30
    args.qp_list = [30, 31, 32, 34, 36, 40, 44, 50]
    # args.qp_list = [50]
    args.qp_list = args.qp_list + [33, 35, 37, 38, 39]
    # args.qp_list = [30]

    # args.qp_list = [20, 27, 28, 30, 32, 34, 35, 36, 38, 40, 46]
    # args.inputs = [
    #     "visdrone/videos/vis_%d" % i for i in [169, 170, 171, 172, 173]
    # ]
    # args.inputs = ["yoda/yoda_%d" % i for i in range(7, 8)]
    # args.inputs = ["dashcam/dashcamcropped_%d" % i for i in range(1, 11)]
    # args.inputs = ["videos/driving_%d" % i for i in range(5)] + [
    #     "videos/dashcamcropped_%d" % i for i in range(1, 11)
    # ]
    args.inputs = ["artifact/dashcamcropped_%d" % i for i in [1]]

    # args.inputs = ["videos/driving_%d" % i for i in range(5)]

    args.inputs = [args.inputs[i] for i in range(len(args.inputs))]
    args.force = False
    args.app = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    # args.app = "Yolo5s"
    # args.app = "EfficientDet"
    # assert attr == "webm"
    args.stats = f"artifact/stats_QP30_thresh7_segmented_FPN"
    # args.stats = "artifact/stats_QP30_thresh3_segment_Yolo"
    # args.stats = "frozen_stats_MLSys/stats_QP30_thresh4_segment_EfficientDet"
    # args.stats = "frozen_stats_MLSys/stats_QP30_thresh3_dashcamcropped_Yolo"
    # args.stats = "frozen_stats_MLSys/stats_QP30_thresh4_dashcamcropped_EfficientDet"
    args.confidence_threshold = 0.7
    args.gt_confidence_threshold = 0.7
    args.smooth_frames = 10

    # args = parser.parse_args()
    main(args)
