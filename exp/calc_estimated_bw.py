import os
import pickle

import torch

x = os.path.getsize("large_dashcam/large_dashcam_1_qp_30.mp4")
y = os.path.getsize("large_dashcam/large_dashcam_1_qp_40.mp4")

# name = "large_dashcam/large_dashcam_1_blackgen_saliencydual_heat_0.3_largeqp_40.mp4"
# rx = pickle.load(open(f"{name}.qp30.mp4.mask", "rb")).mean().item()
# ry = pickle.load(open(f"{name}.qp40.mp4.mask", "rb")).mean().item()

# print(rx * x + ry * y)

name = "large_dashcam/large_dashcam_1_blackgen_saliency_bound_0.05_conv_5.mp4"
rx = pickle.load(open(f"{name}.mask", "rb")).mean().item()
print(rx)
print(rx * x)
