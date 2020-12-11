import os
import subprocess
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

v_list = ["visdrone/videos/vis_%d" % i for i in [169]]
# v_list = [v_list[2]]
lq = 46
hq_list = ["m1"]
upper_bound = 0.85
lower_bound = 0.1
tile = 16
perc = 5
model_name = "fcn_black_vis_172"
conv = 1

for v, hq in product(v_list, hq_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_compressed_blackgen_dual_hq_{hq}_lq_{lq}_ub_{upper_bound}_lb_{lower_bound}.mp4"

    os.system(f"rm -r {output}*")

    subprocess.run(
        [
            "python",
            "compress_blackgen.py",
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
            "--lower_bound",
            f"{upper_bound}",
            "--upper_bound",
            "1.1",
            "--conv_size",
            f"{conv}",
            # "--force_qp",
            # f"{hq}",
        ]
    )

    subprocess.run(
        [
            "python",
            "compress_blackgen.py",
            "-i",
            f"{v}_qp_50.mp4",
            f"{v}_qp_21.mp4",
            "-o",
            f"{output}.{lq}.mp4",
            "-s",
            f"{v}",
            "--tile_size",
            f"{tile}",
            "-p",
            f"maskgen_pths/{model_name}.pth.best",
            "--lower_bound",
            f"{lower_bound}",
            "--upper_bound",
            f"{upper_bound}",
            "--conv_size",
            f"{conv}",
            "--force_qp",
            f"{lq}",
        ]
    )

    os.system(f"python inference_dual.py -i {output}")
    os.system(f"python examine.py -i {output} -g {v}_qp_20.mp4 {v}_qp_21.mp4")

