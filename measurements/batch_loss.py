import os
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
v_list = [
    "visdrone/videos/vis_171",
]
# v_list = [v_list[2]]
base = 42
high = 30
tile = 16
smooth_list = [1]
perc_list = [96]


for v, perc, smooth in product(v_list, perc_list, smooth_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_blackgen_dual_qp_{high}_{base}_perc_{perc}.mp4"
    if not os.path.exists(output):
        os.system(
            f"python compress_loss.py -i {v}_qp_{base}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output}.qp{high}.mp4 --tile_size {tile} --visualize True"
            f" -g {v}_qp_{high}.mp4 --smooth_frames {smooth} --force_qp {high} --percentile {perc}"
        )
        os.system(f"cp {v}_qp_{base}.mp4 {output}.base.mp4")
        os.system(f"python inference_dual.py -i {output}")

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7"
    )

