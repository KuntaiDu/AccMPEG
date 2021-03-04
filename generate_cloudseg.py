import argparse
import logging
import os
import subprocess
from pathlib import Path
from pdb import set_trace

from munch import Munch

gt_qp = 30
qp_list = [40]


def main(args):

    for video_name in args.inputs:
        assert Path(video_name).is_dir()
        video_name = Path(video_name)

        # # generate ground truth
        # output_names = []
        # for quality in quality_list:
        #     input_name = f"{video_name}/%010d.png"
        #     output_name = f"{video_name}_qp_{gt_qp}_{quality}.mp4"
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
        #     + ["-o", f"{video_name}_qp_{gt_qp}_ground_truth.mp4"]
        # )

        apps = [
            "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
            "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
            "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
        ]
        stats = ["stats_DC5", "stats_FPN", "stats_C4"]

        # generate mpeg curve
        for qp in qp_list:
            input_name = f"{video_name}/%010d.png"
            output_name = f"{video_name}_cloudseg_640x360_qp_{qp}.mp4"
            print(f"Generate video for {output_name}")
            # encode_with_qp(input_name, output_name, qp, args)

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_name,
                    "-start_number",
                    "0",
                    "-vf",
                    "scale=640:360",
                    "-qp",
                    f"{qp}",
                    output_name,
                ]
            )

            for app, stat in zip(apps, stats):

                subprocess.run(
                    [
                        "python",
                        "inference.py",
                        "-i",
                        output_name,
                        "--app",
                        app,
                        "--confidence_threshold",
                        "0.7",
                        "--enable_cloudseg",
                        "True",
                    ]
                )

                subprocess.run(
                    [
                        "python",
                        "examine.py",
                        "-i",
                        output_name,
                        "-g",
                        f"{video_name}_qp_{gt_qp}.mp4",
                        "--app",
                        app,
                        "--confidence_threshold",
                        "0.7",
                        "--gt_confidence_threshold",
                        "0.7",
                        "--stats",
                        stat,
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
    args.inputs = ["visdrone/videos/vis_%d" % i for i in range(169, 174)] + [
        "dashcam/dashcam_%d" % i for i in range(1, 11)
    ]
    # args.inputs = ["dashcam/dashcam_%d" % i for i in [2, 5, 6, 8]]
    # args.inputs = ["visdrone/videos/vis_171"]
    args.force = True
    # args.app = "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml"

    # args = parser.parse_args()
    main(args)
