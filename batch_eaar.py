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

# v_list = [
#     "dashcam/dashcam_2",
#     # "visdrone/videos/vis_170",
#     # "visdrone/videos/vis_173",
#     # "visdrone/videos/vis_169",
#     # "visdrone/videos/vis_172",
#     # "visdrone/videos/vis_209",
#     # "visdrone/videos/vis_217",
# ]
v_list = [f"visdrone/videos/vis_{i}" for i in [173]]
# + [
#     "dashcam/dashcam_%d" % i for i in range(1, 11)
# ]
# v_list = ["dashcam/dashcam_%d" % i for i in range(2, 11)]
attr_list = ["FPN"]
# v_list = ["visdrone/videos/vis_171"]
# v_list = [v_list[2]]
# base = 50
# high_list = [30, 34, 38]
tile = 16
conf_list = [0.7]
# high_low_list = [(30, 42), (34, 46), (38, 50)]
high_low_list = [(38, 50)]
gt = 30
# lower_bound_list = [0.3]

for v, attr, conf, (high, base) in product(
    v_list, attr_list, conf_list, high_low_list
):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_blackgen_dual_eaar3_qp_{base}_{high}_conf_{conf}.mp4"
    app = f"COCO-Detection/faster_rcnn_R_101_{attr}_3x.yaml"

    if True:
        # if True:

        subprocess.run(["rm", "-r", output + "*"])

        subprocess.run(
            [
                "python",
                "compress_eaar.py",
                "-i",
                f"{v}_qp_{base}.mp4",
                f"{v}_qp_{high}.mp4",
                "-s",
                f"{v}",
                "-o",
                f"{output}.qp{high}.mp4",
                "--tile_size",
                f"{tile}",
                # "-g",
                # f"{v}_qp_{high}_ground_truth.mp4",
                "--qp",
                f"{high}",
                "--app",
                app,
                "--conf",
                f"{conf}",
            ]
        )

        os.system(f"cp {v}_qp_{base}.mp4 {output}.base.mp4")

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

        os.system(f"python inference.py -i {output} --app {app}")

        os.system(
            f"python examine.py -i {output} -g {v}_qp_{gt}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7 --app {app} --stats stats_FPN_measurement"
        )

        seg_app = "Segmentation/fcn_resnet50"

        os.system(f"python inference.py -i {output} --app {seg_app}")
        os.system(
            f"python examine.py -i {output} -g {v}_qp_{gt}.mp4  --stats stats_fcn50_measurement --app {seg_app}"
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

