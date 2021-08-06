import os
from datetime import datetime
from itertools import product

import yaml

# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]


# v_list = ["visdrone/videos/vis_%d" % i for i in [170, 171]]
# v_list = [
#     "visdrone/videos/vis_171",
# ]
# v_list = ["visdrone/videos/vis_%d" % i for i in range(169, 174)] + [
#     "dashcam/dashcam_%d" % i for i in range(1, 5)
# ]
v_list = ["visdrone/videos/vis_%d" % i for i in [169]]
# v_list = ["dashcam/dashcam_5"]
# v_list = ["visdrone/videos/vis_172"]
# v_list = ["dashcam/dashcam_%d" % i for i in [3]]
# v_list = [v_list[2]]
base_list = [44]
high = 30
tile = 16
smooth_list = [1]
delta_list = [64]

# app_names = [
#     f"COCO-Detection/faster_rcnn_R_101_{attr}_3x.yaml"
#     for attr in ["FPN", "C4", "DC5"]
# ]

attr_list = ["FPN", "C4", "DC5"]
# attr_list = ["FPN"]

now = datetime.now()

for v, delta, smooth, base, attr in product(
    v_list, delta_list, smooth_list, base_list, attr_list
):

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = (
        f"{v}_blackgen_meas2_gtbbox_qp_{high}_delta_{delta}_attr_{attr}.mp4"
    )
    output2 = f"{v}_blackgen_visualize2_dual_gtbbox_qp_{high}_delta_{delta}_attr_{attr}.mp4"
    # if not os.path.exists(output):

    app = f"COCO-Detection/faster_rcnn_R_101_{attr}_3x.yaml"

    if not os.path.exists(output):
        # os.system(
        #     f"python compress_loss.py -i {v}_qp_{base}.mp4 "
        #     f" {v}_qp_{high}.mp4 -s {v} -o {output}.qp{high}.mp4 --tile_size {tile} --visualize True"
        #     f" -g {v}_qp_{high}.mp4 --smooth_frames {smooth} --qp {high} --percentile {perc} --app {app_name}"
        # )
        # os.system(f"cp {v}_qp_{base}.mp4 {output}.base.mp4")
        os.system(
            f"python compress_loss.py -i {v}_qp_{base}.mp4 "
            f" {v}_qp_{high}.mp4 -s {v} -o {output} --tile_size {tile} --visualize True"
            f" -g {v}_qp_{high}.mp4 --qp {high} --app {app} --delta {delta} --num_iterations 4"
        )
    # os.system(f"cp {output} {output2}.qp30.mp4")
    # os.system(f"cp {output}.mask {output2}.qp30.mp4.mask")
    # os.system(f"cp {output}.args {output2}.qp30.mp4.args")
    # os.system(f"cp {v}_qp_{base}.mp4 {output2}.base.mp4")

    # os.system(
    #     f"python inference.py -i {output} --app {app} --visualize_step_size 25"
    # )

    # os.system(
    #     f"python inference.py -i {output2} --app {app} --from_source True --visualize_step_size 25"
    # )

    for inf_attr in attr_list:

        inf_app = f"COCO-Detection/faster_rcnn_R_101_{inf_attr}_3x.yaml"

        os.system(f"python inference.py -i {output} --app {inf_app}")

        os.system(
            f"python examine.py -i {output} -g {v}_qp_{high}.mp4 --gt_confidence_threshold 0.7 --confidence_threshold 0.7 --stats stats_FPN_measurement_{attr} --app {inf_app}"
        )
