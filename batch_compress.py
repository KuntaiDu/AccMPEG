import os
import subprocess
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]
# v_list = ["youtube_videos/dashcam_%d_crop" % (i + 1) for i in range(4)] + [
#     "youtube_videos/trafficcam_%d_crop" % (i + 1) for i in range(4)
# ]

# v_list = ["dashcam/dashcam_%d" % i for i in [2, 5, 6, 8]]
v_list = ["large_dashcam/large_1"]
# v_list = ["dashcam/dashcam_%d" % i for i in [7]]

# v_list = [
#     "visdrone/videos/vis_170",
#     "visdrone/videos/vis_173",
#     "visdrone/videos/vis_169",
#     "visdrone/videos/vis_172",
#     "visdrone/videos/vis_171",
#     # "visdrone/videos/vis_209",
#     # "visdrone/videos/vis_217",
# ]
# v_list = [v_list[2]]
# v_list = ["visdrone/videos/vis_171"]
base = 50
high = 30
tile = 16
model_name = "COCO_full_normalizedsaliency_vgg11_crossthresh"
conv_list = [5]
bound_list = [0.1]
app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"


for v, conv, bound in product(v_list, conv_list, bound_list):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'

    if True:
        # if True:
        os.system(
            f"python generate_mask.py -i {v}_qp_{high}.mp4 -o {v}_compressed.mp4"
            f" --tile_size {tile}  -p maskgen_pths/{model_name}.pth.best"
        )

        os.system(
            f"python compress.py -o {v}_compressed.mp4 -c black_background_compressor -p {conv} --bound {bound} --smooth_frames 30 -s large_dashcam/large_1.mp4 --qp {high}"
        )
    

