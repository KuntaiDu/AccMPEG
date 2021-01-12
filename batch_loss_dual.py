import os
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
v_list = ["visdrone/videos/vis_171"]
# v_list = [v_list[2]]
base = 42
high = 30
tile = 16
bound_list = [1, 2, 5]
smooth_list = [1]


for v, bound, smooth in product(v_list, bound_list, smooth_list):
    lb = 0.3
    ub = bound

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_blackgen_lossdualdiff3_smooth_{smooth}_qp_{high}_{base}_bound_{ub}_{lb}.mp4"
    # if not os.path.exists(output):
    if True:
        os.system(f"rm -r {output}*")
        os.system(
            f"python compress_loss_dual.py -i {v}_qp_51.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output}.qp{high}.mp4 --tile_size {tile} --visualize True"
            f" -g {v}_qp_{high}_ground_truth.mp4 --upper_bound 10000 --lower_bound {ub} --smooth_frames {smooth} --force_qp {high}"
        )
        os.system(
            f"python compress_loss_dual.py -i {v}_qp_51.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output}.qp{base}.mp4 --tile_size {tile} --visualize True"
            f" -g {v}_qp_{high}_ground_truth.mp4 --upper_bound {ub} --lower_bound {lb} --smooth_frames {smooth} --force_qp {base}"
        )
        os.system(f"python inference_dual.py -i {output}")

    os.system(f"python examine.py -i {output} -g {v}_qp_{high}_ground_truth.mp4")

