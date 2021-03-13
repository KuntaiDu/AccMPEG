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
# v_list = [f"visdrone/videos/vis_{i}" for i in [169, 170, 171, 172, 173]] + [
#     "dashcam/dashcam_%d" % i for i in range(1, 11)
# ]

# v_list = ["dashcam/dashcam_%d" % i for i in range(2, 11)]
v_list = ["visdrone/videos/vis_%d" % i for i in [171, 169, 170, 172, 173]]
attr_list = ["FPN"]
# v_list = ["visdrone/videos/vis_171"]
# v_list = [v_list[2]]
base_high_list = [(40, 30), (44, 34), (48, 38)]
# high = 30
gt = 30
tile = 16

ext_list = ["mp4"]
# lower_bound_list = [0.3]

# conf_list = [0.9, 0.8, 0.6]


for ext, v, attr, (base, high) in product(
    ext_list, v_list, attr_list, base_high_list
):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_blackgen_dual_dds2_qp_{base}_{high}_{attr}_conf_0.8_tile_{tile}.{ext}"
    app = f"COCO-Detection/faster_rcnn_R_101_{attr}_3x.yaml"

    # if True:
    #     # if True:

    #     subprocess.run(["rm", "-r", output + "*"])

    #     subprocess.run(
    #         [
    #             "python",
    #             "compress_dds.py",
    #             "-i",
    #             f"{v}_qp_{base}.{ext}",
    #             f"{v}_qp_{high}.{ext}",
    #             "-s",
    #             f"{v}",
    #             "-o",
    #             f"{output}.qp{high}.{ext}",
    #             "--tile_size",
    #             f"{tile}",
    #             # "-g",
    #             # f"{v}_qp_{high}_ground_truth.mp4",
    #             "--qp",
    #             f"{high}",
    #             "--app",
    #             app,
    #         ]
    #     )

    #     os.system(f"cp {v}_qp_{base}.{ext} {output}.base.{ext}")

    # os.system(f"python inference.py -i {output} --app {app}")

    # os.system(
    #     f"python examine.py -i {output} -g {v}_qp_{gt}.{ext} --gt_confidence_threshold 0.7 --confidence_threshold 0.7 --app {app} --stats stats_FPN_measurement"
    # )

    seg_app = "Segmentation/fcn_resnet50"

    os.system(f"python inference.py -i {output} --app {seg_app}")
    os.system(
        f"python examine.py -i {output} -g {v}_qp_{gt}.mp4  --stats stats_fcn50_measurement_new --app {seg_app}"
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

