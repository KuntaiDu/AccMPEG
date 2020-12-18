import argparse
import logging
import os
import subprocess
from pathlib import Path

qp = 30
quality_list = ["veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]

# qp_list = [42, 46, 50]
# qp_list = [34]
# qp_list = [20]
# qp_list = [qp_list[0]]


def main(args):

    for video_name in args.inputs:
        assert Path(video_name).is_dir()
        video_name = Path(video_name)

        # os.system(f"rm {video_name}_qp_{qp}_*.mp4")

        output_names = []
        for quality in quality_list:
            input_name = f"{video_name}/%010d.png"
            output_name = f"{video_name}_qp_{qp}_{quality}.mp4"
            output_names.append(output_name)
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
                        "-preset",
                        f"{quality}",
                        output_name,
                    ]
                )

                subprocess.run(["python", "inference.py", "-i", output_name])

        subprocess.run(
            ["python", "merge_ground_truth.py", "-i"]
            + output_names
            + ["-o", f"{video_name}_qp_{qp}_ground_truth.mp4"]
        )

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
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=16
    )
    args = parser.parse_args()
    main(args)
