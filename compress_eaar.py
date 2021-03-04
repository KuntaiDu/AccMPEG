"""
    Compress the video through gradient-based optimization.
"""

import argparse
import gc
import logging
import time
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
from utils.bbox_utils import center_size
from utils.loss_utils import focal_loss as get_loss
from utils.mask_utils import *
from utils.results_utils import read_ground_truth, read_results
from utils.timer import Timer
from utils.video_utils import get_qp_from_name, read_videos, write_video
from utils.visualize_utils import visualize_heat_by_summarywriter

sns.set()


def main(args):

    gc.enable()

    # initialize
    logger = logging.getLogger("eaar")
    logger.addHandler(logging.FileHandler("eaar.log"))
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    videos = videos
    bws = [0, 1]
    qps = [get_qp_from_name(video_name) for video_name in video_names]

    # construct applications
    app = DNN_Factory().get_model(args.app)

    writer = SummaryWriter(f"runs/{args.app}/{args.output}")

    # construct the mask
    mask_shape = [
        len(videos[-1]),
        1,
        720 // args.tile_size,
        1280 // args.tile_size,
    ]
    mask = torch.ones(mask_shape).float()

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

    logger.info(f"Processing application {app.name}")
    progress_bar = enlighten.get_manager().counter(
        total=len(videos[-1]), desc=f"{app.name}", unit="frames"
    )

    losses = []
    f1s = []

    for fid, (video_slices, mask_slice) in enumerate(
        zip(zip(*videos), mask.split(1))
    ):

        progress_bar.update()

        lq_image, hq_image = video_slices[0], video_slices[1]
        # lq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_34/%05d.png' % (fid+offset2)))[None, :, :, :]

        # with Timer("rpn", logger):
        proposals = app.region_proposal(hq_image, detach=True)

        # with Timer("filter", logger):
        proposals = proposals[
            proposals.proposal_boxes.area() < 0.1 * 1280 * 720
        ]
        proposals = proposals[proposals.objectness_logits > args.conf]
        regions = center_size(proposals.proposal_boxes.tensor).cpu()

        # with Timer("generate", logger):
        maskB = generate_mask_from_regions(
            mask_slice.cuda(), regions, 0, args.tile_size, cuda=True
        )
        mask_delta = maskB
        mask_delta[mask_delta < 0] = 0
        mask_slice[:, :, :, :] = mask_delta.cpu()

        # visualization
        if fid % args.visualize_step_size == 0:
            image = T.ToPILImage()(video_slices[-1][0, :, :, :])
            visualize_heat_by_summarywriter(
                image,
                mask_slice.cpu().detach().float(),
                "inferred_saliency",
                writer,
                fid,
                args,
            )

        # plt.clf()
        # sns.distplot(heat.flatten().detach().numpy())
        # plt.savefig(
        #     f"visualize/{args.output}/{fid}_dist.png", bbox_inches="tight"
        # )

    mask.requires_grad = False
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
        "--visualize_step_size", type=int, help="Visualization", default=100,
    )
    parser.add_argument(
        "--app", type=str, help="The name of the model.", required=True,
    )

    parser.add_argument(
        "--conf", type=float, help="The original video source.", default=0.8,
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        help="The video file names. The largest video file will be the ground truth.",
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="The original video source.",
        required=True,
    )
    # parser.add_argument('-g', '--ground_truth', type=str, help='The ground truth results.', required=True)
    parser.add_argument(
        "-o", "--output", type=str, help="The output name.", required=True
    )
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=8
    )
    # parser.add_argument(
    #     "--upper_bound", type=float, help="The upper bound for the mask", required=True,
    # )
    # parser.add_argument(
    #     "--lower_bound", type=float, help="The lower bound for the mask", required=True,
    # )
    parser.add_argument(
        "--confidence_threshold",
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
    parser.add_argument("--qp", type=int, required=True)

    # parser.add_argument('--mask', type=str,
    #                     help='The path of the ground truth video, for loss calculation purpose.', required=True)

    args = parser.parse_args()

    main(args)
