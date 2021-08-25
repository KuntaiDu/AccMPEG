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
v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)]
# v_list = ["visdrone/videos/vis_%d" % i for i in [171]]
# v_list = [v_list[2]]
high = 30
tile = 16
perc = 5
qp_list = [38, 40, 44, 50]

attr_list = ["C4", "DC5"]

# app_name = "Segmentation/fcn_resnet50"


for attr, v, qp in product(attr_list, v_list, qp_list):

    app_name = f"COCO-Detection/faster_rcnn_R_101_{attr}_3x.yaml"

    stats = f"stats_gtbbox_{attr}"

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    orig = f"{v}_compressed_blackgen_obj_{attr}.mp4"
    output = f"{v}_compressed_blackgen_dual_obj_qp_{qp}_{attr}.mp4"

    if not os.path.exists(orig):
        os.system(
            f"python compress_object.py -i {v}_qp_30.mp4 -g {v}_qp_30.mp4 "
            f"-s {v} -o {orig} --qp {high} --app {app_name} --visualize_step_size 1000"
        )
    os.system(f"rm -r {output}*")
    os.system(f"cp {orig} {output}.qp30.mp4")
    # set_trace()
    os.system(f"cp {orig}.mask {output}.qp30.mp4.mask")
    os.system(f"cp {orig}.args {output}.qp30.mp4.args")
    os.system(f"cp {v}_qp_{qp}.mp4 {output}.base.mp4")

    os.system(
        f"python inference.py -i {output} --app {app_name} --visualize_step_size 1000"
    )

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7 --stats {stats} --app {app_name}"
    )

