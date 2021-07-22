import argparse
import logging
import os
import subprocess
from pathlib import Path

from munch import Munch

from utils.results_utils import read_results

gt_qp = 30
qp_list = [30, 50]
attr = "mp4"


def main(args):

    logger = logging.getLogger("mpeg_curve")

    for video_name in args.inputs:

        # generate mpeg curve
        for qp in qp_list:
            output_name = f"{video_name}_qp_{qp}.{attr}"
            print(f"Generate video for {output_name}")
            print(f"Generating {video_name}_{qp}.mp4")

            if not os.path.exists(output_name):
                if attr == "mp4":

                    subprocess.run(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel", "warning",
                            "-stats",
                            "-y",
                            "-i",
                            video_name+f".{attr}",
                            "-start_number",
                            "0",
                            "-qp",
                            f"{qp}",
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
                    "0.7",
                    "--gt_confidence_threshold",
                    "0.7",
                    "--stats",
                    args.stats,
                ]
            )



if __name__ == "__main__":


    args = Munch()
    args.inputs = ["large_dashcam/large_1"]
    args.force = True
    args.app = "Detr_ResNet101"
    args.stats = f"stats"

    main(args)
