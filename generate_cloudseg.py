import argparse
import logging
import os
import subprocess
from pathlib import Path
from pdb import set_trace

from munch import Munch

gt_qp = 30
qp_list = [25]


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

        # generate mpeg curve
        for qp in qp_list:
            input_name = f"{video_name}/%010d.png"
            output_name = f"{video_name}_cloudseg_qp_{qp}.mp4"
            print(f"Generate video for {output_name}")
            # encode_with_qp(input_name, output_name, qp, args)

            if args.force or not os.path.exists(output_name):

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

            subprocess.run(
                [
                    "python",
                    "inference_segmentation.py",
                    "-i",
                    output_name,
                    "--enable_cloudseg",
                    "True",
                ]
            )

            subprocess.run(
                [
                    "python",
                    "examine_segmentation.py",
                    "-i",
                    output_name,
                    "-g",
                    f"{video_name}_qp_30.mp4",
                    "--gt_confidence_threshold",
                    "0.7",
                    "--confidence_threshold",
                    "0.7",
                    "--stats",
                    "stats_segmentation",
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
    args.inputs = ["visdrone/videos/vis_%d" % i for i in [171, 172, 173]]
    # args.inputs = ["dashcam/dashcam_%d" % (i + 1) for i in range(10)] + [
    #     f"visdrone/videos/vis_{i}" for i in [169, 170, 171, 172, 173]
    # ]
    args.force = False

    # args = parser.parse_args()
    main(args)
