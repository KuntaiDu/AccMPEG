import argparse
import logging
import os
import subprocess
from pathlib import Path

qp_list = [20, 21, 22, 23, 24, 26, 28, 32, 34, 38, 42, 46, 50]
# qp_list = [42, 46, 50]
# qp_list = [34]
# qp_list = [20]
# qp_list = [qp_list[0]]


def main(args):

    for video_name in args.inputs:
        assert Path(video_name).is_dir()
        video_name = Path(video_name)
        for qp in qp_list:
            input_name = f"{video_name}/%010d.png"
            output_name = f"{video_name}_qp_{qp}.mp4"
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
                    "-qmin",
                    f"{qp}",
                    "-qmax",
                    f"{qp}",
                    output_name,
                ]
            )

            subprocess.run(["python", "inference.py", "-i", output_name])

        for qp in qp_list:
            output_name = f"{video_name}_qp_{qp}.mp4"
            subprocess.run(
                [
                    "python",
                    "examine.py",
                    "-i",
                    output_name,
                    "-g",
                    f"{video_name}_qp_20.mp4",
                    f"{video_name}_qp_21.mp4",
                ]
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        help="The video file names. The largest video file will be the ground truth.",
        required=True,
    )
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=16
    )
    args = parser.parse_args()
    main(args)
