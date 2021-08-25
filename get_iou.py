import glob
import pickle
from pdb import set_trace

import yaml


def get_iou(gt, obj):

    with open(gt, "rb") as f:
        gt_mask = pickle.load(f)

    with open(obj, "rb") as f:
        obj_mask = pickle.load(f)

    iou = []

    for gt_slice, obj_slice in zip(gt_mask, obj_mask):

        gt_slice = gt_slice == 1
        obj_slice = obj_slice == 1

        iou.append(
            ((gt_slice & obj_slice).sum() / (gt_slice | obj_slice).sum()).item()
        )
    return iou


attr = "FPN"
folder = "visdrone/videos"
gt_fmt = f"vis_%d*meas2_gtbbox*delta_64*{attr}*.mask"
obj_fmt = f"vis_%d*blackgen_obj_{attr}.mp4.mask"


def get_name(fmt, fid):
    print(fmt)
    ret = glob.glob1(folder, fmt % fid)
    print(ret)
    assert len(ret) == 1
    return folder + "/" + ret[0]


result = []

for fid in range(169, 174):
    result = result + get_iou(get_name(gt_fmt, 171), get_name(obj_fmt, 171))


with open(f"IoU_{attr}.txt", "w") as f:
    f.write(yaml.dump(result))
