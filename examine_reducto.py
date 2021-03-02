import argparse
import json
import logging
import os
import pickle
import subprocess
from pathlib import Path
from pdb import set_trace

import coloredlogs
import enlighten
import networkx as nx
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision import io

from dnn.dnn_factory import DNN_Factory
from utils.bbox_utils import jaccard
from utils.results_utils import read_results, write_results
from utils.video_utils import read_bandwidth


def main(args):

    logger = logging.getLogger("examine")
    handler = logging.NullHandler()
    logger.addHandler(handler)

    app = DNN_Factory().get_model(args.app)
    fids = json.load(open(args.json, "r"))

    # inference first
    Path(f"{args.input}.source.pngs").mkdir(exist_ok=True)
    Path(f"{args.input}.pngs").mkdir(exist_ok=True)
    new_fid = 0
    for new_fid, fid in enumerate(fids):
        print(fid)
        subprocess.run(
            [
                "cp",
                f"{args.source}/%010d.png" % fid,
                f"{args.input}.source.pngs/%010d.png" % new_fid,
            ]
        )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            f"{args.input}.source.pngs/%010d.png",
            "-start_number",
            "0",
            "-qp",
            "28",
            f"{args.input}",
        ]
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            f"{args.input}",
            "-start_number",
            "0",
            f"{args.input}.pngs/%010d.png",
        ]
    )

    for _ in range(1):

        ground_truth_results = read_results(args.ground_truth, app.name, logger)

        video_results = {}

        for new_fid, fid in enumerate(fids):
            print(fid)
            image = Image.open(f"{args.input}.pngs/%010d.png" % new_fid)
            image = T.ToTensor()(image)[None, :, :, :].cuda()
            video_results[fid] = app.inference(image, detach=True)

        last_fid = 0

        for fid in ground_truth_results.keys():
            if fid in video_results:
                last_fid = fid
                print(last_fid)
            else:
                video_results[fid] = video_results[last_fid]

        metrics = app.calc_accuracy(video_results, ground_truth_results, args)
        res = {
            "application": app.name,
            "video_name": args.input,
            "bw": os.path.getsize(args.input),
            "ground_truth_name": args.ground_truth,
        }
        res.update(metrics)
        with open(args.stats, "a") as f:
            f.write(yaml.dump([res]))


if __name__ == "__main__":

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--stats", type=str, default="stats")

    parser.add_argument("--json", type=str, required=True)
    parser.add_argument(
        "-g",
        "--ground_truth",
        type=str,
        help="The ground-truth video name.",
        required=True,
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="The confidence score threshold for calculating accuracy.",
        default=0.7,
    )
    parser.add_argument(
        "--app", type=str, help="The name of the model.", required=True,
    )
    parser.add_argument(
        "--gt_confidence_threshold",
        type=float,
        help="The confidence score threshold for calculating accuracy.",
        default=0.7,
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        help="The IoU threshold for calculating accuracy in object detection.",
        default=0.5,
    )
    parser.add_argument("--input", type=str, required=True)

    args = parser.parse_args()

    main(args)
