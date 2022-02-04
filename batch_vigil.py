import os
from itertools import product
from pdb import set_trace

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ["videos/dashcamcropped_%d" % i for i in range(3, 8)] + [
#     "videos/driving_%d" % i for i in range(5)
# ]
v_list = ["videos/dashcamcropped_%d" % i for i in range(7, 8)]
# v_list = ["dashcam/dashcam_5"]

# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
# v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)] + ['dashcam/da']
# v_list = ["visdrone/videos/vis_%d" % i for i in [171]]
# v_list = [v_list[2]]
high = 30
tile = 16
perc = 5
base = 51


vigil_app_name = "MobileNet-SSD"


# stats = "frozen_stats_MLSys/stats_QP30_thresh7_segmented_FPN"
# conf_thresh = 0.7
# gt_conf_thresh = 0.7
# app_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

stats = "frozen_stats_MLSys/stats_QP30_thresh3_segment_Yolo"
conf_thresh = 0.3
gt_conf_thresh = 0.3
app_name = "Yolo5s"

# stats = "frozen_stats_MLSys/stats_QP30_thresh4_segment_EfficientDet"
# conf_thresh = 0.4
# gt_conf_thresh = 0.4
# app_name = "EfficientDet"

visualize_step_size = 10000


for v in v_list:

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    orig = f"{v}_vigil_hq_{high}_lq_{base}.mp4"

    if not os.path.exists(orig):

        # os.system(
        #     f"python inference.py -i {v}_qp_30.mp4 --app {vigil_app_name} --confidence_threshold 0.0"
        # )
        os.system(
            f"python compress_object.py -i {v}_qp_30.mp4 -g {v}_qp_30.mp4 "
            f"-s {v} -o {orig} --hq {high} --lq {base} --app {vigil_app_name} --visualize_step_size 1000 --confidence_threshold 0.0 --gt_confidence_threshold 0.0"
        )

    output = orig

    os.system(
        f"python inference.py -i {output} --app {app_name} --confidence_threshold {conf_thresh} --gt_confidence_threshold {gt_conf_thresh} --visualize_step_size {visualize_step_size} "
        # f" --visualize --lq_result {v}_qp_{base}.mp4 --ground_truth {v}_qp_{high}.mp4"
    )

    os.system(
        f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --confidence_threshold {conf_thresh}  --gt_confidence_threshold {gt_conf_thresh} --app {app_name} --stats {stats}"
    )
