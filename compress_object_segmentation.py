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
    logger = logging.getLogger("object_seg")
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
    mask = torch.ones(mask_shape).float()

    ground_truth_dict = read_results(args.ground_truth, app.name, logger)

    writer = SummaryWriter(f"runs/{args.app}/{args.output}")

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

    for fid, (video_slices, mask_slice) in enumerate(
        zip(zip(*videos), mask.split(1))
    ):

        progress_bar.update()
        # lq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_34/%05d.png' % (fid+offset2)))[None, :, :, :]

        gt = (ground_truth_dict[fid] > 0).float()[:, None, :, :]
        gt = F.conv2d(
            gt,
            torch.ones(1, 1, args.tile_size, args.tile_size),
            stride=args.tile_size,
        )
        gt = (gt > 0.5).float()

        mask[fid : fid + 1, :, :, :] = gt

        if fid % args.visualize_step_size == 0:

            image = T.ToPILImage()(video_slices[-1][0, :, :, :])

            writer.add_image("raw_frame", video_slices[-1][0, :, :, :], fid)

            visualize_heat_by_summarywriter(
                image,
                mask_slice.cpu().detach().float(),
                "inferred_saliency",
                writer,
                fid,
                args,
            )

    mask = dilate_binarize(mask, 0.5, args.conv_size, cuda=False)

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
        "--visualize_step_size", type=int, help="Visualization", default=100,
    )
    parser.add_argument(
        "--conv_size", type=int, help="Visualization", required=True,
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
    parser.add_argument(
        "--visualize",
        type=bool,
        help="Visualize the mask if True",
        default=True,
    )
    parser.add_argument("--hq", type=int, required=True)
    parser.add_argument("--lq", type=int, required=True)
    parser.add_argument(
        "--app", type=str, help="The name of the model.", required=True,
    )
    parser.add_argument(
        "--smooth_frames",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=10,
    )

    args = parser.parse_args()

    main(args)
