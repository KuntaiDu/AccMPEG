import math

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Ellipse

stats = yaml.safe_load(open("stats_QP30_thresh7_segmented_FPN", "r").read())

# color palatte
colors = [
    "#c19277",  # mpeg
    "#62959c",  # ours
    "#9dad7f",  # dds
    "#d9dab0",  # eaar
    "#a98b98",  # cloudseg
    "#c1c0b9",  # reducto
]

name2color = {
    "mpeg": "#c19277",  # mpeg
    "accmpeg": "#62959c",  # ours
    "dds": "#9dad7f",  # dds
    "eaar": "#d9dab0",  # eaar
    "cloudseg": "#a98b98",  # cloudseg
    "reducto": "#c1c0b9",  # reducto
    "vigil": "#8c96c6",
}

# set default visualization parameters
plt.style.use("ggplot")
plt.rcParams["font.size"] = 30
plt.rc("font", family="sans-serif")
plt.rcParams["font.weight"] = "medium"
plt.rcParams["pdf.fonttype"] = 42


def savefig(filename, fig):
    import time

    timestr = time.strftime("%Y%m%d-%H%M%S")
    fig.savefig(f"{filename}.jpg", bbox_inches="tight")


def get_delay(x):

    RTT = 0.1

    # grant
    streaming_delay = RTT + x["bw"] * 8 / (0.5 * 1e6) / 180

    encoding_delay = 0.131
    if "reducto" in x["video_name"]:
        encoding_delay = 0.0435

    if "roi" in x["video_name"]:
        encoding_delay += 0.08

    if "dds" in x["video_name"] or "eaar" in x["video_name"]:
        # a round trip
        encoding_delay += RTT

    if "dds" in x["video_name"]:
        encoding_delay += RTT + 0.131

    if "reducto" in x["video_name"]:
        # reducto logic runs for 25ms
        encoding_delay += 0.247

    if "vigil" in x["video_name"]:
        # vigil runs for 0.17s cause it operates on lower resolution
        encoding_delay += 0.17

    x["streaming_delay"] = streaming_delay
    x["encoding_delay"] = encoding_delay
    x["delay"] = streaming_delay + encoding_delay


for x in stats:
    get_delay(x)

fig, ax = plt.subplots(figsize=(10, 7))

awstream = [i for i in stats if "dashcamcropped_1_qp_" in i["video_name"]]
accmpeg = [i for i in stats if "roi" in i["video_name"]]

ax.scatter(
    [i["delay"] for i in awstream],
    [i["f1"] for i in awstream],
    label="AWStream",
    c=name2color["mpeg"],
    s=200,
)
ax.scatter(
    [i["delay"] for i in accmpeg],
    [i["f1"] for i in accmpeg],
    label="AccMPEG",
    c=name2color["accmpeg"],
    s=200,
)

ax.set_xlabel("Delay (s)")
ax.set_ylabel("Accuracy")
ax.set_xlim(left=0)
ax.legend()

savefig("delay-accuracy", fig)
