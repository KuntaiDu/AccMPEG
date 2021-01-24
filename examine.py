import argparse
import logging
import pickle
from pathlib import Path
from pdb import set_trace

import coloredlogs
import enlighten
import networkx as nx
import torch
import yaml
from torchvision import io

from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from utils.bbox_utils import jaccard
from utils.results_utils import read_results, write_results
from utils.video_utils import read_bandwidth


def main(args):

    logger = logging.getLogger("examine")
    handler = logging.NullHandler()
    logger.addHandler(handler)

    bws = [read_bandwidth(video) for video in args.inputs]
    video_names = args.inputs

    application_bundle = [FasterRCNN_ResNet50_FPN()]

    for application in application_bundle:

        ground_truth_results = read_results(args.ground_truth, application.name, logger)

        for video_name, bw in zip(video_names, bws):
            video_results = read_results(video_name, application.name, logger)
            metrics = application.calc_accuracy(
                video_results, ground_truth_results, args
            )
            res = {
                "application": application.name,
                "video_name": video_name,
                "bw": bw,
                "ground_truth_name": args.ground_truth,
            }
            res.update(metrics)
            with open("stats", "a") as f:
                f.write(yaml.dump([res]))


if __name__ == "__main__":

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--inputs",
        type=str,
        help="The video file names to obtain inference results.",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        type=str,
        help="The ground-truth video name.",
        required=True,
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="The confidence score threshold for calculating accuracy.",
        default=0.5,
    )
    parser.add_argument(
        "--gt_confidence_threshold",
        type=float,
        help="The confidence score threshold for calculating accuracy.",
        default=0.5,
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        help="The IoU threshold for calculating accuracy in object detection.",
        default=0.5,
    )

    args = parser.parse_args()

    main(args)
