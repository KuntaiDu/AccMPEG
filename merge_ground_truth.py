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
from utils.results_utils import merge_results, read_results, write_results
from utils.video_utils import read_bandwidth


def main(args):

    logger = logging.getLogger("merge")
    handler = logging.NullHandler()
    logger.addHandler(handler)

    # bws = [read_bandwidth(video) for video in args.inputs]
    video_names = args.inputs

    application_bundle = [FasterRCNN_ResNet50_FPN()]

    for application in application_bundle:

        ground_truth_results = [
            read_results(vname, application.name, logger) for vname in args.inputs
        ]
        ground_truth_results = merge_results(ground_truth_results, application, args)
        write_results(args.output, application.name, ground_truth_results, logger)


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
        "-o", "--output", type=str, help="The output pseudo video name.", required=True,
    )
    parser.add_argument(
        "--confidence_threshold",
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
