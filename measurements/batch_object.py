import os
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)]
# v_list = [v_list[2]]
base = 50
high = 24
tile = 16
perc = 5
conv_list = [1]
delta_list = [-1]
qp_list = [36, 32, 30, 28]


for v, conv, delta, qp in product(v_list, conv_list, delta_list, qp_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    orig = f"{v}_compressed_blackgen_object.mp4"
    output = f"{v}_compressed_blackgen_dual_object_qp_{qp}"
    # if len(glob.glob(output + "*")) == 0:
    if True or not os.path.exists(output):
        os.system(f"rm -r {output}*")
        os.system(f"cp {orig} {output}.qp30.mp4")
        os.system(f"cp {orig}.mask {output}.qp30.mp4.mask")
        os.system(f"cp {orig}.args {output}.qp30.mp4.args")
        os.system(f"cp {v}_qp_{qp}.mp4 {output}.base.mp4")

        os.system(f"python inference_dual.py -i {output}")

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7"
    )

