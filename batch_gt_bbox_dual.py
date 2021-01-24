import glob
import os
import subprocess
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
v_list = ["visdrone/videos/vis_172"]
# v_list = [v_list[2]]
base = 50
high = 30
tile = 16
perc = 5
conv_list = [3]
low_list = [50]


for v, conv, low in product(v_list, conv_list, low_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_compressed_blackgen_dual_gt_bbox_qp_{high}_{low}_conv_{conv}.mp4"
    if len(glob.glob(output + "*")) == 0:
        # if True:
        os.system(f"rm -r {output}*")
        os.system(
            f"python compress_gt_bbox.py -i {v}_qp_{base}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output}.qp{high}.mp4 --tile_size {tile}  "
            f" --tile_percentage {perc} --conv_size {conv} --visualize True"
            f" -g {v}_qp_{high}_ground_truth.mp4 --force_qp {high}"
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                f"{v}/%010d.png",
                "-start_number",
                "0",
                "-qp",
                f"{low}",
                # "-vf",
                # "scale=480:272",
                f"{output}.base.mp4",
            ]
        )

        os.system(f"python inference_dual.py -i {output}")

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --gt_confidence_threshold 0.1"
    )

    # os.system(f"python examine.py -i {output} -g {v}_qp_{high}_ground_truth.mp4")

