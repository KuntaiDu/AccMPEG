import os
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)]
# v_list = ["visdrone/videos/vis_171"]
# v_list = [v_list[2]]
base = 42
high = 30
tile = 16
perc = 5
conv_list = [1]
delta_list = [64]


for v, conv, delta in product(v_list, conv_list, delta_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_nonsense.mp4"
    # if len(glob.glob(output + "*")) == 0:
    if True or not os.path.exists(output):
        os.system(f"rm -r {output}*")
        os.system(
            f"python compress_high_perc.py -i {v}_qp_{base}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output} --tile_size {tile}  "
            f" -g {v}_qp_{high}.mp4 --num_iterations 5 --delta {delta}"
        )

