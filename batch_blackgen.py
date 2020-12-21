import os
import subprocess
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170]]
v_list = ["visdrone/videos/vis_169_3rdhalf"]
# v_list = [v_list[2]]
base = 50
high = 22
tile = 16
perc = 5
model_name = "gt_vis_172_cross_entropy_weight_1"
conv_list = [3, 5, 7]
bound_list = [0.5]


for v, conv, bound in product(v_list, conv_list, bound_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = (
        f"{v}_compressed_blackgen_{model_name}_conv_{conv}_bound_{bound}_qp_{high}.mp4"
    )
    gt_output = f"{v}_compressed_blackgen_gt_bbox_conv_{conv}.mp4"

    # if not os.path.exists(output):
    if True:
        os.system(
            f"python compress_blackgen.py -i {v}_qp_{base}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output} --tile_size {tile}  -p maskgen_pths/{model_name}.pth.best"
            f" --conv_size {conv} --visualize True"
            f" -g {v}_qp_{high}_ground_truth.mp4 --bound 0.5"
        )
        os.system(f"python inference.py -i {output}")

    os.system(f"python examine.py -i {output} -g {v}_qp_{high}_ground_truth.mp4")

    if not os.path.exists(f"diff/{output}.gtdiff.mp4"):
        subprocess.run(
            [
                "python",
                "diff.py",
                "-i",
                output,
                gt_output,
                "-o",
                f"diff/{output}.gtdiff.mp4",
            ]
        )

