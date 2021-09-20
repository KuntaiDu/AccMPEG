import os
import subprocess
from itertools import product

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ['train_first/trafficcam_%d_train' % (i+1) for i in range(4)] + ['train_first/dashcam_%d_train' % (i+1) for i in range(4)]
# v_list = [v_list[4]]
app = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# model_name = f"COCO_full_normalizedsaliency_R_101_FPN_crossthresh_5xdownsample"
architecture = "SSD"
model_name = f"COCO_detection_FPN_{architecture}_single"
# model_name = "visdrone_R_101_FPN_crossthresh"
filename = "mobilenet_v2_single_channel"

gt = "pickles/COCO_saliency_FPN.pickle"


subprocess.run(
    [
        "python",
        "train_cityscape_single_channel.py",
        "-g",
        # f"visdrone_normalizedsaliency_R_101_FPN.pickle",
        f"{gt}",
        "-p",
        f"maskgen_pths/{model_name}.pth",
        # "--init",
        # f"maskgen_pths/{model_name}.pth.best",
        "--tile_size",
        "16",
        "--batch_size",
        "4",
        "--log",
        f"train_{model_name}_small.log",
        "--maskgen_file",
        f"/tank/kuntai/code/video-compression/maskgen/{filename}.py",
        "--visualize",
        "True",
        "--app",
        # f"Segmentation/fcn_resnet50",
        f"{app}",
        "--local_rank",
        "1",
        "--num_workers",
        "10",
        "--learning_rate",
        "1e-3",
        "--architecture",
        architecture,
        "--test_set",
        "object_detection_test_set",
    ]
)
