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
v_list = ["visdrone/videos/vis_%d" % i for i in [169, 170, 171, 172, 173]]
# v_list = ["visdrone/videos/vis_%d" % i for i in [171]]
# v_list = [v_list[2]]
high = 30
tile = 16
perc = 5
qp_list = [40]
# app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
app_name = "Segmentation/fcn_resnet50"
conv_list = [1]

dual = True


for v, qp, conv in product(v_list, qp_list, conv_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    orig = f"{v}_compressed_blackgen_segmentation_dds_conv_{conv}.mp4"
    # output = f"{v}_compressed_blackgen_dual_obj_qp_{qp}_conv_{conv}.mp4"

    if not os.path.exists(orig):
        # os.system(f"python inference.py -i {v}_qp_30.mp4 --app {app_name}")
        os.system(
            f"python compress_object_segmentation.py -i {v}_qp_30.mp4 -g {v}_qp_30.mp4 "
            f"-s {v} -o {orig} --qp {high} --app {app_name} --conv_size {conv}"
        )

    if dual:

        output = f"{v}_compressed_blackgen_dual_segmentation_dds_qp_{qp}_cv_{conv}.mp4"

        os.system(f"rm -r {output}*")
        os.system(f"cp {orig} {output}.qp30.mp4")
        # set_trace()
        os.system(f"cp {orig}.mask {output}.qp30.mp4.mask")
        os.system(f"cp {orig}.args {output}.qp30.mp4.args")
        os.system(f"cp {v}_qp_{qp}.mp4 {output}.base.mp4")

        orig = output

    os.system(f"python inference.py -i {orig} --app {app_name}")

    os.system(
        f"python examine.py -i {orig} -g {v}_qp_{high}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7 --stats sstats_segmentation_final --app {app_name}"
    )

