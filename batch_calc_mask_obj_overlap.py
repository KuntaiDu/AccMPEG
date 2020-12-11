import os
import subprocess
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

v_list = ["visdrone/videos/vis_%d" % i for i in [172]]
# v_list = [v_list[2]]
hq_list = [21]
bound_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
# bound_list = [0.8]
tile = 16
perc = 5
model_name = "fcn_black_vis_172"

for v, hq, bound in product(v_list, hq_list, bound_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_compressed_calc_{bound}.mp4"

    os.system(f"rm -r {output}*")

    subprocess.run(
        [
            "python",
            "calc_mask_obj_overlap.py",
            "-i",
            f"{v}_qp_50.mp4",
            f"{v}_qp_21.mp4",
            "-o",
            f"{output}.{hq}.mp4",
            "-s",
            f"{v}",
            "--tile_size",
            f"{tile}",
            "-p",
            f"maskgen_pths/{model_name}.pth.best",
            "--bound",
            f"{bound}",
            "--force_qp",
            f"{hq}",
            "--visualize",
            "True",
        ]
    )

    # os.system(f"python inference_dual.py -i {output}")
    # os.system(f"python examine.py -i {output} -g {v}_qp_20.mp4 {v}_qp_21.mp4")

