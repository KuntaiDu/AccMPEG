import argparse
import logging
import os
import subprocess
from pathlib import Path

gt_qp = 22
qp_list = [22, 24, 26, 28, 30, 34, 38, 42, 46, 50]
quality_list = ["veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]


def main(args):

    for video_name in args.inputs:
        assert Path(video_name).is_dir()
        video_name = Path(video_name)

        # generate ground truth
        output_names = []
        for quality in quality_list:
            input_name = f"{video_name}/%010d.png"
            output_name = f"{video_name}_qp_{gt_qp}_{quality}.mp4"
            output_names.append(output_name)
            print(f"Generate video for {output_name}")
            # encode_with_qp(input_name, output_name, qp, args)

            if not os.path.exists(output_name):
                # encode
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        input_name,
                        "-start_number",
                        "0",
                        "-qmin",
                        f"{gt_qp}",
                        "-qmax",
                        f"{gt_qp}",
                        "-preset",
                        f"{quality}",
                        output_name,
                    ]
                )
                # and inference
                subprocess.run(["python", "inference.py", "-i", output_name])

        subprocess.run(
            ["python", "merge_ground_truth.py", "-i"]
            + output_names
            + ["-o", f"{video_name}_qp_{gt_qp}_ground_truth.mp4"]
        )

        # generate mpeg curve
        for qp in qp_list:
            input_name = f"{video_name}/%010d.png"
            output_name = f"{video_name}_qp_{qp}.mp4"
            print(f"Generate video for {output_name}")
            # encode_with_qp(input_name, output_name, qp, args)

            if not os.path.exists(output_name):

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
                    f"{video_name}_qp_30_ground_truth.mp4",
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
        "-f",
        "--force",
        type=bool,
        help="Force the program to regenerate all the outputs or not.",
        default=False,
    )

    args = parser.parse_args()
    main(args)
