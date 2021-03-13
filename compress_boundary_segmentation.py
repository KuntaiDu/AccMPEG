"""
    Compress the video through gradient-based optimization.
"""

import argparse
import gc
import logging
import subprocess
from pathlib import Path

import coloredlogs
import enlighten
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import io

from dnn.dnn_factory import DNN_Factory
from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from maskgen.fcn_16_single_channel import FCN
from utils.bbox_utils import center_size
from utils.loss_utils import focal_loss as get_loss
from utils.mask_utils import *
from utils.results_utils import read_ground_truth, read_results
from utils.video_utils import get_qp_from_name, read_videos, write_video
from utils.visualize_utils import visualize_heat_by_summarywriter

sns.set()


def error_tile(mask):

    maxval = mask.max().item()

    ret = torch.zeros_like(mask)

    for i in range(1, maxval + 1):
        val = (mask == i).int()
        val = F.conv2d(
            val, torch.ones([1, 1, i * 2 + 1, i * 2 + 1]).int(), padding=i
        )
        ret = ret + val

    return (ret > 0.5).int()


def main(args):

    gc.enable()

    # initialize
    logger = logging.getLogger("meas")
    # logger.addHandler(logging.FileHandler("blackgen.log"))
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    videos = videos
    bws = [0, 1]
    qps = [get_qp_from_name(video_name) for video_name in video_names]

    # construct applications
    app = DNN_Factory().get_model(args.app)

    # mask_generator = FCN()
    # mask_generator.load(args.path)
    # mask_generator.train().cuda()

    # construct the mask
    mask_shape = [
        len(videos[-1]),
        1,
        720 // args.tile_size,
        1280 // args.tile_size,
    ]
    error = torch.zeros(mask_shape).int()

    ground_truth_dict = read_results(args.ground_truth, app.name, logger)
    # low_quality_dict = read_results(args.base, app.name, logger)

    writer = SummaryWriter(f"runs/{args.app}/{args.output}")

    logger.info(f"Processing application {app.name}")
    progress_bar = enlighten.get_manager().counter(
        total=len(videos[-1]), desc=f"{app.name}", unit="frames"
    )

    for iteration in range(args.num_iterations + 1):

        tps = []

        logger.info(f"Processing application {app.name}")
        progress_bar = enlighten.get_manager().counter(
            total=len(videos[-1]), desc=f"{app.name}", unit="frames"
        )

        for fid, (video_slices, error_slice) in enumerate(
            zip(zip(*videos), error.split(1))
        ):

            progress_bar.update()

            # construct hybrid image, lq: normalized color, hq: hq image
            lq_image, hq_image = video_slices[0], video_slices[1]
            mask_slice = error_tile(error_slice).float()
            mask_tile = tile_mask(mask_slice, args.tile_size)
            # set_trace()
            mix_image = lq_image * (1 - mask_tile) + hq_image * mask_tile

            # get result
            result = app.inference(mix_image.cuda(), detach=True)

            delta = (result != ground_truth_dict[fid]).int()[:, None, :, :]
            delta = F.conv2d(
                delta,
                torch.ones([1, 1, args.tile_size, args.tile_size]).int(),
                stride=args.tile_size,
            )
            delta = (delta > 0.5).int()
            error_slice[:, :, :, :] = error_slice + delta

            tps += [
                app.calc_accuracy(
                    {fid: result}, {fid: ground_truth_dict[fid]}, args
                )["acc"]
            ]

            # logger.info("f1: %.3f", tps[-1])
            # logger.info(
            #     "Perc: %.3f", (mask_slice == 1).sum() / (mask_slice >= 0).sum()
            # )

        logger.info("Average TP: %.3f", torch.tensor(tps).mean().item())
        logger.info(
            "Average mask: %.3f", error_tile(error).float().mean().item()
        )

    mask = error_tile(error).float()

    write_black_bkgd_video_smoothed_continuous(
        mask, args, args.qp, logger, writer=writer, tag="hq"
    )
    # masked_video = generate_masked_video(mask, videos, bws, args)
    # write_video(masked_video, args.output, logger)


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
        nargs="+",
        help="The video file names. The largest video file will be the ground truth.",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        help="The video file names. The largest video file will be the ground truth.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-b",
        "--base",
        help="The video file names. The largest video file will be the ground truth.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="The original video source.",
        required=True,
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        help="Number of iterations needed",
        default=1,
    )
    # parser.add_argument('-g', '--ground_truth', type=str, help='The ground truth results.', required=True)
    parser.add_argument(
        "-o", "--output", type=str, help="The output name.", required=True
    )
    parser.add_argument(
        "--visualize_step_size", type=int, help="Visualization", default=100,
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="The confidence score threshold for calculating accuracy.",
        default=0.7,
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
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=16
    )
    # parser.add_argument(
    #     "--conv_size", type=int, help="The tile size of the mask.", default=0
    # )
    parser.add_argument(
        "--visualize",
        type=bool,
        help="Visualize the mask if True",
        default=True,
    )
    parser.add_argument("--qp", type=int, required=True)
    parser.add_argument(
        "--app", type=str, help="The name of the model.", required=True,
    )

    args = parser.parse_args()

    main(args)
