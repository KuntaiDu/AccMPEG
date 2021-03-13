import os
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)] + [
#     "dashcam/dashcam_%d" % i for i in range(1, 11)
# ]

v_list = ["large_object/bus2", "large_object/bus"]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
# v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)]
# v_list = ["visdrone/videos/vis_%d" % i for i in [171]]
# v_list = [v_list[2]]
high = 30
tile = 16
perc = 5
qp_list = [44]
conv_list = [1]
gt = 30

assert high == gt
# app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
app_name = "Segmentation/fcn_resnet101"


for v, qp, conv in product(v_list, qp_list, conv_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    orig = f"{v}_compressed_blackgen_error.mp4"
    output = f"{v}_compressed_blackgen_dual_error_qp_{qp}_{high}.mp4"

    if True or not os.path.exists(orig):
        os.system(
            f"python compress_boundary_segmentation.py -i {v}_qp_{qp}.mp4 {v}_qp_{high}.mp4 -g {v}_qp_{gt}.mp4 -b {v}_qp_{qp}.mp4 "
            f"-s {v} -o {orig} --qp {high} --app {app_name} --visualize_step_size 10"
        )
    os.system(f"rm -r {output}*")
    os.system(f"cp {orig} {output}.qp{high}.mp4")
    os.system(f"cp {orig}.mask {output}.qp{high}.mp4.mask")
    os.system(f"cp {orig}.args {output}.qp{high}.mp4.args")
    os.system(f"cp {v}_qp_{qp}.mp4 {output}.base.mp4")

    os.system(
        f"python inference.py -i {output} --app {app_name} --visualize_step_size 10 --from_source True"
    )

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{gt}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7 --stats stats_measurement_segmentation_large_object --app {app_name}"
    )

    os.system(
        f"python inference.py -i {output} --app {app_name} --visualize_step_size 10"
    )

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{gt}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7 --stats stats_measurement_segmentation_large_object --app {app_name}"
    )

