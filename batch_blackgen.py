import os
import subprocess
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]
# v_list = ["youtube_videos/dashcam_%d_crop" % (i + 1) for i in range(4)] + [
#     "youtube_videos/trafficcam_%d_crop" % (i + 1) for i in range(4)
# ]

# v_list = ["dashcam/dashcam_%d" % i for i in [2, 5, 6, 8]]
v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)] + [
    "dashcam/dashcam_%d" % i for i in range(1, 11)
]
# v_list = ["dashcam/dashcam_%d" % i for i in [7]]

# v_list = [
#     "visdrone/videos/vis_170",
#     "visdrone/videos/vis_173",
#     "visdrone/videos/vis_169",
#     "visdrone/videos/vis_172",
#     "visdrone/videos/vis_171",
#     # "visdrone/videos/vis_209",
#     # "visdrone/videos/vis_217",
# ]
# v_list = [v_list[2]]
# v_list = ["visdrone/videos/vis_171"]
base = 50
high = 30
tile = 16
model_name = "COCO_full_normalizedsaliency_R_101_DC5_crossthresh"
conv_list = [5, 9, 13]
bound_list = [0.2]

app_name = "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml"


for v, conv, bound in product(v_list, conv_list, bound_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_blackgen_bound_{bound}_qp_{high}_conv_{conv}.mp4"

    if not os.path.exists(output):
        # if True:
        os.system(
            f"python compress_blackgen.py -i {v}_qp_{base}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output} --tile_size {tile}  -p maskgen_pths/{model_name}.pth.best"
            f" --conv_size {conv} "
            f" -g {v}_qp_{high}.mp4 --bound {bound} --qp {high} --smooth_frames 30 --app {app_name}"
        )
        os.system(f"python inference.py -i {output} --app {app_name}")

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --confidence_threshold 0.7 --gt_confidence_threshold 0.7 --app {app_name}"
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

