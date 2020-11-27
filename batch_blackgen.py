import os
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

v_list = ["visdrone/videos/vis_%d" % i for i in [170]]
# v_list = [v_list[2]]
base = 34
tile = 16
perc = 5
model_name = "fcn_black_vis_172"
thresh = 0.8

for v in v_list:

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_compressed_blackgen_{model_name}_thresh_{thresh}.hevc"

    os.system(
        f"python compress_blackgen.py -i {v}_qp_{base}.hevc "
        f" {v}_qp_22.hevc -s {v}.yuv -o {output} --tile_size {tile}  -p maskgen_pths/{model_name}.pth.best"
        f" --tile_percentage {perc} --mask_threshold {thresh}"
    )
    os.system(f"python inference.py -i {output}")
    os.system(f"python examine.py -i {output} -g {v}_qp_22.hevc")

