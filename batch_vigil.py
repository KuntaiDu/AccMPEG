import os
from itertools import product
from pdb import set_trace

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)] + [
    "dashcam/dashcam_%d" % i for i in range(1, 11)
]

# v_list = ["dashcam/dashcam_5"]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
# v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)] + ['dashcam/da']
# v_list = ["visdrone/videos/vis_%d" % i for i in [171]]
# v_list = [v_list[2]]
high = 30
tile = 16
perc = 5

attr_list = ["FPN", "C4", "DC5"]

app_name = "MobileNet-SSD"


for attr, v in product(attr_list, v_list):

    examine_app_name = f"COCO-Detection/faster_rcnn_R_101_{attr}_3x.yaml"

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    orig = f"{v}_compressed_blackgen_vigil.mp4"

    if not os.path.exists(orig):
        os.system(
            f"python inference.py -i {v}_qp_30.mp4 --app {app_name} --confidence_threshold 0.0"
        )
        os.system(
            f"python compress_object.py -i {v}_qp_30.mp4 -g {v}_qp_30.mp4 "
            f"-s {v} -o {orig} --qp {high} --app {app_name} --visualize_step_size 1000 --confidence_threshold 0.0 --gt_confidence_threshold 0.0"
        )

    os.system(
        f"python inference.py -i {orig} --app {examine_app_name} --visualize_step_size 1000"
    )

    stats = f"stats_{attr}"

    os.system(
        f"python examine.py -i {orig} -g {v}_qp_{high}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7 --stats {stats} --app {examine_app_name}"
    )

