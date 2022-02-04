import os
from itertools import product
from pdb import set_trace

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)] + [
#     "dashcam/dashcam_%d" % i for i in range(1, 11)
# ]

# v_list = ["dashcam/dashcam_5"]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
v_list = ["videos/surf_%d_final" % i for i in [4]]
# v_list = ["visdrone/videos/vis_%d" % i for i in [171]]
# v_list = [v_list[2]]
high = 30
tile = 16
base_qp_list = [44, 40, 36]

# app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
app_name = "Segmentation/fcn_resnet50"
conv_list = [3]
stats = "stats_temp"


for base, conv, v in product(v_list, base_qp_list, conv_list, v_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    orig = f"{v}_dds_conv_{conv}.mp4"
    # output = f"{v}_compressed_blackgen_dual_obj_qp_{qp}_conv_{conv}.mp4"

    if not os.path.exists(orig):
        # os.system(f"python inference.py -i {v}_qp_30.mp4 --app {app_name}")
        os.system(
            f"python compress_object_segmentation.py -i {v}_qp_{base}.mp4 -g {v}_qp_{high}.mp4 "
            f"-s {v} -o {orig} --hq {high} --lq {base} --app {app_name} --conv_size {conv}"
        )

    os.system(f"python inference.py -i {orig} --app {app_name}")

    os.system(
        f"python examine.py -i {orig} -g {v}_qp_{high}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7 --stats {stats} --app {app_name}"
    )

