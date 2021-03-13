seg_fmt = "visdrone/videos/vis_%d_compressed_blackgen_obj_conv_13.mp4"
det_fmt = (
    "visdrone/videos/vis_%d_blackgen_meas2_gtbbox_qp_30_delta_64_attr_FPN.mp4"
)

seg_app = "Segmentation/fcn_resnet50"
det_app = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

vids = [171, 170, 172, 173, 169]

import os

for vid in vids:

    seg = seg_fmt % vid
    det = det_fmt % vid

    os.system(f"python inference.py -i {seg} --app {det_app}")
    os.system(f"python inference.py -i {det} --app {seg_app}")

    os.system(
        f"python examine.py -i {seg} -g visdrone/videos/vis_{vid}_qp_30.mp4 --app {seg_app} --stats stats_fig6"
    )
    os.system(
        f"python examine.py -i {seg} -g visdrone/videos/vis_{vid}_qp_30.mp4 --app {det_app} --stats stats_fig6"
    )
    os.system(
        f"python examine.py -i {det} -g visdrone/videos/vis_{vid}_qp_30.mp4 --app {seg_app} --stats stats_fig6"
    )
    os.system(
        f"python examine.py -i {det} -g visdrone/videos/vis_{vid}_qp_30.mp4 --app {det_app} --stats stats_fig6"
    )
