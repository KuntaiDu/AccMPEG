import os
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)] + [
    "dashcam/dashcam_%d" % i for i in range(1, 11)
]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
# v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)]
# v_list = ["visdrone/videos/vis_%d" % i for i in [171]]
# v_list = [v_list[2]]
high = 30
tile = 16
perc = 5
qp_list = [44, 40, 36]
# app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
app_name = "Segmentation/fcn_resnet50"


for v, qp in product(v_list, qp_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    orig = f"{v}_compressed_blackgen_obj.mp4"
    output = f"{v}_compressed_blackgen_dual_obj_qp_{qp}"

    # if not os.path.exists(orig):
    #     os.system(
    #         f"python compress_object.py -i {v}_qp_30.mp4 -g {v}_qp_30.mp4 "
    #         f"-s {v} -o {orig} --qp {high} --app {app_name}"
    #     )
    # os.system(f"rm -r {output}*")
    # os.system(f"cp {orig} {output}.qp30.mp4")
    # os.system(f"cp {orig}.mask {output}.qp30.mp4.mask")
    # os.system(f"cp {orig}.args {output}.qp30.mp4.args")
    # os.system(f"cp {v}_qp_{qp}.mp4 {output}.base.mp4")

    os.system(f"python inference.py -i {output} --app {app_name}")

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7 --stats stats_fcn50_measurement --app {app_name}"
    )

