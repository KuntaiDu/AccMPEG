import argparse
import logging
import os
import subprocess
from pathlib import Path

qp = 30


# qp_list = [42, 46, 50]
# qp_list = [34]
# qp_list = [20]
# qp_list = [qp_list[0]]


def main(args):

    for video_name in args.inputs:
        assert Path(video_name).is_dir()
        video_name = Path(video_name)

        # os.system(f"rm {video_name}_qp_{qp}_*.mp4")

        # for quality in quality_list:
        #     x = f"{video_name}_qp_{qp}_{quality}.mp4"
        #     y = f"{video_name}_qp_{qp}_ground_truth.mp4"
        #     subprocess.run(
        #         ["python", "diff.py", "-i", x, y, "-o", "diff/" + x + ".gtdiff"]
        #     )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        help="The video file names. The largest video file will be the ground truth.",
        required=True,
    )
    args = parser.parse_args()
    main(args)
