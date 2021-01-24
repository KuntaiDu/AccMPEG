import glob
import os
import subprocess
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]
# v_list = ["youtube_videos/dashcam_%d_crop" % (i + 1) for i in range(4)] + [
#     "youtube_videos/trafficcam_%d_crop" % (i + 1) for i in range(4)
# ]

# v_list = ["youtube_videos/dashcam_%d_crop" % (i + 1) for i in range(4)]
# v_list = ["dashcam/dashcam_2"]
# v_list = ["visdrone/videos/vis_172"]

v_list = [
    "visdrone/videos/vis_171",
    # "visdrone/videos/vis_170",
    # "visdrone/videos/vis_173",
    # "visdrone/videos/vis_169",
    # "visdrone/videos/vis_172",
    # "visdrone/videos/vis_209",
    # "visdrone/videos/vis_217",
]
# v_list = [v_list[2]]
base = 42
high = 30
tile = 16
model_name = "COCO_full_normalizedsaliency_vgg11_crossthresh"
conv_large_list = [13]
conv_list = [7]
bound_list = [0.3]
# lower_bound_list = [0.3]

for v, conv, bound, conv_large in product(
    v_list, conv_list, bound_list, conv_large_list
):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_final_blackgen_dualconv_bound_{bound}_conv_{conv}_{conv_large}.mp4"

    if True or len(glob.glob(output + "*.mp4")) == 0:
        # if True:

        subprocess.run(["rm", "-r", output + "*"])

        subprocess.run(
            [
                "python",
                "compress_blackgen_dual.py",
                "-i",
                f"{v}_qp_{base}.mp4",
                f"{v}_qp_{high}.mp4",
                "-s",
                f"{v}",
                "-o",
                f"{output}.qp{base}.mp4",
                "--tile_size",
                f"{tile}",
                "-p",
                f"maskgen_pths/{model_name}.pth.best",
                "--conv_size",
                f"{conv}",
                "--conv_size_large",
                f"{conv_large}",
                "--visualize",
                "True",
                # "-g",
                # f"{v}_qp_{high}_ground_truth.mp4",
                "--bound",
                f"{bound}",
                "--force_qp",
                f"{base}",
                "--smooth_frames",
                "30",
            ]
        )

        subprocess.run(
            [
                "python",
                "compress_blackgen_dual.py",
                "-i",
                f"{v}_qp_{base}.mp4",
                f"{v}_qp_{high}.mp4",
                "-s",
                f"{v}",
                "-o",
                f"{output}.qp{high}.mp4",
                "--tile_size",
                f"{tile}",
                "-p",
                f"maskgen_pths/{model_name}.pth.best",
                "--conv_size",
                f"{conv}",
                "--conv_size_large",
                "-1",
                "--visualize",
                "True",
                # "-g",
                # f"{v}_qp_{high}_ground_truth.mp4",
                "--bound",
                f"{bound}",
                "--force_qp",
                f"{high}",
                "--smooth_frames",
                "30",
            ]
        )

        # subprocess.run(
        #     [
        #         "ffmpeg",
        #         "-y",
        #         "-i",
        #         f"{v}/%010d.png",
        #         "-start_number",
        #         "0",
        #         "-qp",
        #         f"30",
        #         "-vf",
        #         "scale=480:272",
        #         f"{output}.base.mp4",
        #     ]
        # )

        os.system(f"python inference_dual.py -i {output}")

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7"
    )

    # if not os.path.exists(f"diff/{output}.gtdiff.mp4"):
    #     gt_output = f"{v}_compressed_blackgen_gt_bbox_conv_{conv}.mp4"
    #     subprocess.run(
    #         [
    #             "python",
    #             "diff.py",
    #             "-i",
    #             output,
    #             gt_output,
    #             "-o",
    #             f"diff/{output}.gtdiff.mp4",
    #         ]
    #     )

