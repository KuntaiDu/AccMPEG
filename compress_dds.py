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
from detectron2.structures.boxes import pairwise_iou
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import io

from dnn.dnn_factory import DNN_Factory
from utilities.bbox_utils import center_size
from utilities.compressor import h264_roi_compressor_segment
from utilities.loss_utils import focal_loss as get_loss
from utilities.mask_utils import *
from utilities.results_utils import read_ground_truth, read_results
from utilities.timer import Timer
from utilities.video_utils import get_qp_from_name, read_videos, write_video
from utilities.visualize_utils import visualize_heat_by_summarywriter

sns.set()


def main(args):

    gc.enable()

    # initialize
    logger = logging.getLogger("dds")
    logger.addHandler(logging.FileHandler("dds.log"))
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    videos = videos
    bws = [0, 1]
    qps = [get_qp_from_name(video_name) for video_name in video_names]

    # construct applications
    app = DNN_Factory().get_model(args.app)

    writer = SummaryWriter(f"runs/{args.app}/{args.output}", flush_secs=10)

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

        try:

            progress_bar.update()

            lq_image, _ = video_slices[0], video_slices[1]
            # lq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_34/%05d.png' % (fid+offset2)))[None, :, :, :]

            lq_inference = app.inference(lq_image, detach=True)
            lq_inference = app.filter_result(lq_inference, args, gt=False)

            proposals = app.region_proposal(lq_image, detach=True)
            proposals = proposals[proposals.objectness_logits > args.conf]
            proposals = proposals[
                proposals.proposal_boxes.area() < 0.02 * 1280 * 720
            ]
            # filter out regions that overlaps with inference results
            iou = pairwise_iou(
                proposals.proposal_boxes, lq_inference["instances"].pred_boxes
            )
            iou = iou > 0.3
            iou = iou.sum(dim=1)
            proposals = proposals[iou == 0]
            regions = center_size(proposals.proposal_boxes.tensor).cpu()

            # boxes = center_size(lq_inference["instances"].pred_boxes.tensor).cpu()

            # maskA = generate_mask_from_regions(
            #     mask_slice.clone(), boxes, 0, args.tile_size
            # )
            maskB = generate_mask_from_regions(
                mask_slice.cuda(), regions, 0, args.tile_size, cuda=True
            )
            mask_delta = maskB
            mask_delta[mask_delta < 0] = 0
            mask_slice[:, :, :, :] = mask_delta

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

        except (IndexError, TypeError):
            mask_slice[:, :, :, :] = 0
            continue

        # plt.clf()
        # sns.distplot(heat.flatten().detach().numpy())
        # plt.savefig(
        #     f"visualize/{args.output}/{fid}_dist.png", bbox_inches="tight"
        # )

    mask.requires_grad = False
    mask = dilate_binarize(mask, 0.5, 3, cuda=False)
    # write_black_bkgd_video_smoothed_continuous(
    #     mask, args, args.qp, logger, writer=writer, tag="hq"
    # )

    mask = (mask > 0.5).int()
    mask = torch.where(
        mask == 1,
        args.hq * torch.ones_like(mask),
        args.lq * torch.ones_like(mask),
    )

    h264_roi_compressor_segment(mask, args, logger)
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
        "-i",
        "--inputs",
        nargs="+",
        help="The video file names. The largest video file will be the ground truth.",
        required=True,
    )
    parser.add_argument(
        "--smooth_frames",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=10,
    )
    parser.add_argument(
        "--app", type=str, help="The name of the model.", required=True,
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="The original video source.",
        required=True,
    )
    parser.add_argument(
        "--conf", type=float, help="The original video source.", default=0.7,
    )
    parser.add_argument(
        "-o", "--output", type=str, help="The output name.", required=True
    )
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=8
    )
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

    parser.add_argument("--hq", type=int, required=True)
    parser.add_argument("--lq", type=int, required=True)

    # parser.add_argument('--mask', type=str,
    #                     help='The path of the ground truth video, for loss calculation purpose.', required=True)

    args = parser.parse_args()

    main(args)
