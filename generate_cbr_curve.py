import argparse
import logging
import os
import subprocess
from pathlib import Path

from munch import Munch

from utils.results_utils import read_results

gt_qp = 6
qp_list = [
    6,
    1,
    2,
    3,
    4,
    5,
]
# qp_list = [32, 42]
quality_list = [
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]


def main(args):

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
        for qp in qp_list:
            input_name = f"{video_name}/%010d.png"
            output_name = f"{video_name}_cbr_{qp}.mp4"
            print(f"Generate video for {output_name}")
            # encode_with_qp(input_name, output_name, qp, args)

            if args.force or not os.path.exists(output_name):

                # subprocess.run(
                #     [
                #         "ffmpeg",
                #         "-y",
                #         "-i",
                #         input_name,
                #         "-start_number",
                #         "0",
                #         "-c:v",
                #         "libx265",
                #         "-x265-params",
                #         f"qp={qp}",
                #         output_name,
                #     ]
                # )

                # ffmpeg -i input.mp4 -c:v libx264 -x264-params "nal-hrd=cbr" -b:v 1M -minrate 1M -maxrate 1M -bufsize 2M output.ts

                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        input_name,
                        "-start_number",
                        "0",
                        "-c:v",
                        "libx264",
                        "-x264-params",
                        "nal-hrd=cbr",
                        "-b:v",
                        f"{qp}M",
                        "-minrate",
                        f"{qp}M",
                        "-maxrate",
                        f"{qp}M",
                        "-bufsize",
                        "2M",
                        output_name,
                    ]
                )

                # subprocess.run(
                #     [
                #         "kvazaar",
                #         "-i",
                #         input_name,
                #         "--input-res",
                #         "1280x720",
                #         "-q",
                #         f"{qp}",
                #         "-o",
                #         output_name,
                #     ]
                # )

                # try:
                #     from pdb import set_trace

                #     # set_trace()
                #     result = read_results(output_name, args.app, logger)
                # except FileNotFoundError:
                subprocess.run(
                    [
                        "python",
                        "inference.py",
                        "-i",
                        output_name,
                        "--app",
                        args.app,
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
                    f"{video_name}_cbr_{gt_qp}.mp4",
                    "--app",
                    args.app,
                    "--confidence_threshold",
                    "0.7",
                    "--gt_confidence_threshold",
                    "0.7",
                    "--stats",
                    args.stats,
                ]
            )

        # for qp in qp_list:
        #     output_name = f"{video_name}_qp_{qp}.mp4"
        #     subprocess.run(
        #         [
        #             "python",
        #             "examine.py",
        #             "-i",
        #             output_name,
        #             "-g",
        #             f"{video_name}_qp_30_ground_truth.mp4",
        #         ]
        #     )


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "-i",
    #     "--inputs",
    #     nargs="+",
    #     help="The video file names. The largest video file will be the ground truth.",
    #     required=True,
    # )
    # parser.add_argument(
    #     "-f",
    #     "--force",
    #     type=bool,
    #     help="Force the program to regenerate all the outputs or not.",
    #     default=False,
    # )

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
    args.inputs = ["adapt/drive_%d" % i for i in range(60)]
    # args.inputs = ["dashcam/dashcam_%d" % i for i in range(4, 11)]
    # args.inputs = ["dashcam/dashcam_%d" % i for i in [2, 5, 6, 8]]
    # args.inputs = ["visdrone/videos/vis_171"]
    args.force = True
    args.app = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    # args.app = "Segmentation/fcn_resnet50"
    args.stats = "stats_adapt_drive"

    # args = parser.parse_args()
    main(args)
