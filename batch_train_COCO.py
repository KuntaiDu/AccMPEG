import os
import subprocess
from itertools import product

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ['train_first/trafficcam_%d_train' % (i+1) for i in range(4)] + ['train_first/dashcam_%d_train' % (i+1) for i in range(4)]
# v_list = [v_list[4]]

model_name = "COCO_full_normalizedsaliency_R_101_DC5_crossthresh"
filename = "vgg11"

subprocess.run(
    [
        "python",
        "train_COCO.py",
        "-g",
        f"COCO_full_normalizedsaliency_R_101_DC5.pickle",
        "-p",
        f"maskgen_pths/{model_name}.pth",
        "--init",
        f"maskgen_pths/COCO_full_normalizedsaliency_R_101_DC5_crossthresh.pth.best",
        "--tile_size",
        "16",
        "--batch_size",
        "2",
        "--log",
        f"train_{model_name}.log",
        "--maskgen_file",
        f"/tank/kuntai/code/video-compression/maskgen/{filename}.py",
        "--visualize",
        "True",
        "--app",
        "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
    ]
)
