import os
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
v_list = ["DAVIS/videos/DAVIS_1"]
# v_list = [v_list[2]]
base = 40
high = 30
tile = 16
perc = 5
large_qp_list = [42]
# model_name = "fcn_black_vis_172"
conv_list = [5]
bound_list = [0.03]
smooth_list = [30]
app = "Segmentation/deeplabv3_resnet50"
bound_large_qp_list = [0.01]
lowres_qp = 40


for v, conv, bound, smooth, bound_large_qp, large_qp in product(
    v_list,
    conv_list,
    bound_list,
    smooth_list,
    bound_large_qp_list,
    large_qp_list,
):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f"{v}_blackgen_saliencydualprotectfocal_bound_{bound}_{bound_large_qp}_lqp_{large_qp}_conv_{conv}_{1}.mp4"
    if not os.path.exists(output):

        os.system(f"rm -r {output}*")

        os.system(
            f"python compress_saliency_dual.py -i {v}_288p_qp_{lowres_qp}.mp4 {v}_qp_{large_qp}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output} --tile_size {tile}  "
            f" --conv_size {conv} "
            f" -g {v}_qp_{high}.mp4 --bound_qp {bound} --bound_large_qp {bound_large_qp} --smooth_frames {smooth} --qp {high} --large_qp {large_qp} --app {app}"
        )
        os.system(f"python inference.py -i {output} --app {app}")

    os.system(f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --app {app}")

