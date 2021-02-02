import os
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
v_list = ["large_dashcam/large_dashcam_1"]
# v_list = [v_list[2]]
base = 50
high = 30
tile = 16
perc = 5
# model_name = "fcn_black_vis_172"
conv_list = [5]
bound_list = [0.01]
smooth_list = [1]
app = "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml"


for v, conv, bound, smooth in product(v_list, conv_list, bound_list, smooth_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_blackgen_saliency_bound_{bound}_conv_{conv}.mp4"
    if not os.path.exists(output):

        os.system(
            f"python compress_saliency.py -i {v}_qp_{base}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output} --tile_size {tile}  "
            f" --conv_size {conv} "
            f" -g {v}_qp_{high}.mp4 --bound {bound} --smooth_frames {smooth} --force_qp {high} --app {app}"
        )
        os.system(f"python inference.py -i {output} --app {app}")

    os.system(f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --app {app}")

