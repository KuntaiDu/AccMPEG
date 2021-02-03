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
from torchvision import io

from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from maskgen.fcn_16_single_channel import FCN
from utils.bbox_utils import center_size
from utils.loss_utils import focal_loss as get_loss
from utils.mask_utils import *
from utils.results_utils import read_ground_truth, read_results
from utils.video_utils import get_qp_from_name, read_videos, write_video

sns.set()


def main(args):

    gc.enable()

    # initialize
    logger = logging.getLogger("blackgen")
    logger.addHandler(logging.FileHandler("blackgen.log"))
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    videos = videos
    bws = [0, 1]
    qps = [get_qp_from_name(video_name) for video_name in video_names]
    qps = [min(qps)]
    if args.force_qp:
        qps = [args.force_qp]

    # construct applications
    application = FasterRCNN_ResNet50_FPN()

    # mask_generator = FCN()
    # mask_generator.load(args.path)
    # mask_generator.train().cuda()

    # construct the mask
    mask_shape = [len(videos[-1]), 1, 720 // args.tile_size, 1280 // args.tile_size]
    mask = torch.ones(mask_shape).float()

    ground_truth_results = read_results(
        args.ground_truth, "FasterRCNN_ResNet50_FPN", logger
    )

    regions = [
        center_size(application.filter_results(i, args.confidence_threshold)[2])
        for i in ground_truth_results.values()
    ]

    for region in regions:
        region[:, 2:] = 0
    # logger.info('Reading ground truth mask')
    # with open(args.mask + '.mask', 'rb') as f:
    #     ground_truth_mask = pickle.load(f)
    # ground_truth_mask = ground_truth_mask[sorted(ground_truth_mask.keys())[1]]
    # ground_truth_mask = ground_truth_mask.split(1)

    plt.clf()
    plt.figure(figsize=(16, 10))

    # binarized_mask = mask.clone().detach()
    # binarize_mask(binarized_mask, bws)
    # if iteration > 3 * (args.num_iterations // 4):
    #     (args.binarize_weight * torch.tensor(iteration*1.0) * (binarized_mask - mask).abs().pow(2).mean()).backward()

    for iteration in range(args.num_iterations + 1):

        logger.info(f"Processing application {application.name}")
        progress_bar = enlighten.get_manager().counter(
            total=len(videos[-1]), desc=f"{application.name}", unit="frames"
        )

        write_black_bkgd_video_smoothed_continuous(mask, args, qps, bws, logger)
        subprocess.run(["python", "inference.py", "-i", args.output])
        inference_results = read_results(args.output, application.name, logger)

        metric = application.calc_accuracy(
            inference_results, ground_truth_results, args
        )

        logger.info(
            "F1: %.3f, Pr: %.3f, Re: %.3f", metric["f1"], metric["pr"], metric["re"]
        )

        for fid in inference_results:

            progress_bar.update()
            # lq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_34/%05d.png' % (fid+offset2)))[None, :, :, :]

            # construct hybrid image
            index = application.get_undetected_ground_truth_index(
                ground_truth_results[fid], inference_results[fid], args
            )

            if iteration == args.num_iterations:
                regions[fid][index, 2:] = 0
            else:
                regions[fid][index, 2:] += args.delta

            mask[fid : fid + 1, :, :, :] = generate_mask_from_regions(
                mask[fid : fid + 1, :, :, :], regions[fid], 0, args.tile_size
            )

    write_black_bkgd_video_smoothed_continuous(mask, args, qps, bws, logger)
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
        "--num_iterations", type=int, default=10,
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "-s", "--source", type=str, help="The original video source.", required=True
    )
    parser.add_argument("--delta", type=int, default=32)
    # parser.add_argument('-g', '--ground_truth', type=str, help='The ground truth results.', required=True)
    parser.add_argument(
        "-o", "--output", type=str, help="The output name.", required=True
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
        "--tile_size", type=int, help="The tile size of the mask.", default=8
    )
    parser.add_argument("--bound", type=float, default=0.5)
    parser.add_argument("--smooth_frames", type=int, default=1)
    # parser.add_argument(
    #     "-p",
    #     "--path",
    #     type=str,
    #     help="The path of pth file that stores the generator parameters.",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--upper_bound", type=float, help="The upper bound for the mask", required=True,
    # )
    # parser.add_argument(
    #     "--lower_bound", type=float, help="The lower bound for the mask", required=True,
    # )
    parser.add_argument(
        "--visualize", type=bool, help="Visualize the mask if True", default=False,
    )
    parser.add_argument("--conv_size", type=int, required=True)
    parser.add_argument("--force_qp", type=int, default=-1)

    # parser.add_argument('--mask', type=str,
    #                     help='The path of the ground truth video, for loss calculation purpose.', required=True)

    args = parser.parse_args()

    main(args)
