import os
import subprocess
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170]]
<<<<<<< HEAD
v_list = [
    "visdrone/videos/vis_171",
    "visdrone/videos/vis_170",
    "visdrone/videos/vis_173",
    "visdrone/videos/vis_169",
    "visdrone/videos/vis_172",
]
=======
v_list = ["visdrone/videos/vis_170", "visdrone/videos/vis_173"]
>>>>>>> 93c028ba893c3eeffc6b513f0a76e17451c150ad
# v_list = [v_list[2]]
base = 50
high = 30
tile = 16
<<<<<<< HEAD
model_name = "COCO_normalizedsaliency_vgg11"
conv_list = [3]
bound_list = [0.5]
=======
model_name = "saliency_vis_172_cross_entropy"
conv_list = [1]
bound_list = [0.1]
>>>>>>> 93c028ba893c3eeffc6b513f0a76e17451c150ad


for v, conv, bound in product(v_list, conv_list, bound_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
<<<<<<< HEAD
    output = f"{v}_blackgen2_{model_name}_bound_{bound}_qp_{high}_conv_{conv}.mp4"

    # if not os.path.exists(output):
    if True:
=======
    output = f"{v}_compressed_blackgen_{model_name}_bound_{bound}_qp_{high}.mp4"

    if not os.path.exists(output):
        # if True:
>>>>>>> 93c028ba893c3eeffc6b513f0a76e17451c150ad
        os.system(
            f"python compress_blackgen.py -i {v}_qp_{base}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output} --tile_size {tile}  -p maskgen_pths/{model_name}.pth.best"
            f" --conv_size {conv} --visualize True"
<<<<<<< HEAD
            f" -g {v}_qp_{high}_ground_truth.mp4 --bound {bound} --force_qp {high} --smooth_frames 30"
=======
            f" -g {v}_qp_{high}_ground_truth.mp4 --bound {bound} --force_qp {high}"
>>>>>>> 93c028ba893c3eeffc6b513f0a76e17451c150ad
        )
        os.system(f"python inference.py -i {output}")

    os.system(f"python examine.py -i {output} -g {v}_qp_{high}_ground_truth.mp4")

    # if not os.path.exists(f"diff/{output}.gtdiff.mp4"):
    #     gt_output = f"{v}_compressed_blackgen_gt_bbox_conv_{conv}.mp4"
    #     subprocess.run(
    #         [
    #             "python",
    #             "diff.py",
    #             "-i",
    #             output,
    #             gt_output,
    #             "-o",
    #             f"diff/{output}.gtdiff.mp4",
    #         ]
    #     )

