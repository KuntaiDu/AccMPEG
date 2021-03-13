import pickle

import torch
import yaml

# mask_names = [
#     "visdrone/videos/vis_%d_blackgen_bound_0.5_qp_30_conv_9_app_FPN.mp4.mask"
#     % i
#     for i in range(169, 174)
# ]

mask_names = [
    "dashcam/dashcam_%d_blackgen_bound_0.5_qp_30_conv_9_app_FPN.mp4.mask" % i
    for i in range(1, 8)
]

thresh = 0.85
ub = 102


def IoU(maskA, maskB):
    assert (maskA == maskB).sum() / maskA.numel() <= 1
    return (maskA == maskB).sum() / maskA.numel()


rights = []

for mask_name in mask_names:

    print(mask_name)

    with open(mask_name, "rb") as f:
        mask = pickle.load(f)

    for i in range(len(mask)):

        right = i
        for j in range(i + 1, len(mask)):

            if IoU(mask[i], mask[j]) > thresh:
                right = j
            else:
                break

            if right - i > ub:
                break
        rights = rights + [right - i]

    print(torch.tensor(rights).float().mean())

print(torch.tensor(rights).float().mean())

with open("persistance_dashcam.txt", "w") as f:
    yaml.dump(rights, f)


import pickle

import torch
import yaml

mask_names = [
    "visdrone/videos/vis_%d_blackgen_bound_0.5_qp_30_conv_9_app_FPN.mp4.mask"
    % i
    for i in range(169, 174)
]

# mask_names = [
#     "dashcam/dashcam_%d_blackgen_bound_0.5_qp_30_conv_9_app_FPN.mp4.mask" % i
#     for i in range(1, 8)
# ]


def IoU(maskA, maskB):
    assert (maskA == maskB).sum() / maskA.numel() <= 1
    return (maskA == maskB).sum() / maskA.numel()


rights = []

for mask_name in mask_names:

    print(mask_name)

    with open(mask_name, "rb") as f:
        mask = pickle.load(f)

    for i in range(len(mask)):

        right = i
        for j in range(i + 1, len(mask)):

            if IoU(mask[i], mask[j]) > thresh:
                right = j
            else:
                break

            if right - i > ub:
                break
        rights = rights + [right - i]

    print(torch.tensor(rights).float().mean())

print(torch.tensor(rights).float().mean())

with open("persistance_drone.txt", "w") as f:
    yaml.dump(rights, f)
