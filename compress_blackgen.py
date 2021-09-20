"""
    Compress the video through gradient-based optimization.
"""

import argparse
import gc
import importlib
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
from tqdm import tqdm

from dnn.dnn_factory import DNN_Factory
from utils.bbox_utils import center_size
from utils.loss_utils import focal_loss as get_loss
from utils.mask_utils import *
from utils.results_utils import read_ground_truth, read_results
from utils.timer import Timer
from utils.video_utils import get_qp_from_name, read_videos, write_video
from utils.visualize_utils import (
    visualize_dist_by_summarywriter,
    visualize_heat_by_summarywriter,
)

thresh_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]

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

    # construct applications
    app = DNN_Factory().get_model(args.app)

    maskgen_spec = importlib.util.spec_from_file_location(
        "maskgen", args.maskgen_file
    )
    maskgen = importlib.util.module_from_spec(maskgen_spec)
    maskgen_spec.loader.exec_module(maskgen)
    mask_generator = maskgen.FCN()
    mask_generator.load(args.path)
    # mask_generator.eval()
    mask_generator.cuda()

    # construct the mask
    mask_shape = [
        len(videos[-1]),
        1,
        720 // args.tile_size,
        1280 // args.tile_size,
    ]
    mask = torch.ones(mask_shape).float()

    # construct the writer for writing the result
    writer = SummaryWriter(f"runs/{args.app}/{args.output}")

    for temp in range(1):

        logger.info(f"Processing application")
        progress_bar = enlighten.get_manager().counter(
            total=len(videos[-1]), desc=f"{app.name}", unit="frames"
        )

        # application.cuda()

        losses = []
        f1s = []

        for fid, (video_slices, mask_slice) in enumerate(
            zip(zip(*videos), mask.split(1))
        ):

            progress_bar.update()

            lq_image, hq_image = video_slices[0], video_slices[1]
            # lq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_34/%05d.png' % (fid+offset2)))[None, :, :, :]

            # construct hybrid image
            with torch.no_grad():
                # gt_result = application.inference(hq_image.cuda(), detach=True)[0]
                # _, _, boxes, _ = application.filter_results(
                #     gt_result, args.confidence_threshold
                # )
                # boxes = center_size(boxes)

                # size1 = boxes[:, 2] * boxes[:, 3]
                # sum1s.append(size1.sum())
                # boxes[:, 2:] = boxes[:, 2:] + 7 * args.tile_size
                # size2 = boxes[:, 2] * boxes[:, 3]
                # sum2s.append(size2.sum())
                # # ratios.append(size2.sum() / size1.sum())
                # mask_slice[:, :, :, :] = generate_mask_from_regions(
                #     mask_slice, boxes, 0, args.tile_size
                # )

                # mask_gen = mask_generator(
                #     torch.cat([hq_image, hq_image - lq_image], dim=1).cuda()
                # )
                hq_image = hq_image.cuda()
                # mask_generator = mask_generator.cpu()
                # with Timer("maskgen", logger):
                mask_gen = mask_generator(hq_image)
                # losses.append(get_loss(mask_gen, ground_truth_mask[fid]))
                mask_gen = mask_gen.softmax(dim=1)[:, 1:2, :, :]
                # mask_lb = dilate_binarize(mask_gen, args.bound, args.conv_size)
                # mask_ub = dilate_binarize(mask_gen, args.upper_bound, args.conv_size)
                mask_slice[:, :, :, :] = mask_gen
                # mask_slice[:, :, :, :] = torch.where(mask_gen > 0.5, torch.ones_like(mask_gen), torch.zeros_like(mask_gen))

            # visualization
            if fid % args.visualize_step_size == 0:

                image = T.ToPILImage()(video_slices[-1][0, :, :, :])

                mask_slice = mask_slice.detach().cpu()

                writer.add_image("raw_frame", video_slices[-1][0, :, :, :], fid)

                visualize_heat_by_summarywriter(
                    image, mask_slice, "inferred_saliency", writer, fid, args,
                )

                visualize_dist_by_summarywriter(
                    mask_slice, "saliency_dist", writer, fid,
                )

                mask_slice = sum(
                    [(mask_slice > thresh).float() for thresh in thresh_list]
                )

                visualize_heat_by_summarywriter(
                    image, mask_slice, "binarized_saliency", writer, fid, args,
                )

        logger.info("In video %s", args.output)
        logger.info("The average loss is %.3f" % torch.tensor(losses).mean())

        # application.cpu()

    mask.requires_grad = False

    for mask_slice in tqdm(mask.split(args.smooth_frames)):

        # mask_slice[:, :, :, :] = (
        #     mask_slice[0:1, :, :, :] + mask_slice[-1:, :, :, :]
        # ) / 2
        mask_slice[:, :, :, :] = mask_slice.mean(dim=0, keepdim=True)

    # if args.bound is not None:
    #     mask = dilate_binarize(mask, args.bound, args.conv_size, cuda=False)
    # else:
    #     assert args.perc is not None
    #     mask = (mask > percentile(mask, args.perc)).float()
    #     mask = dilate_binarize(mask, 0.5, args.conv_size, cuda=False)
    if args.bound is not None:
        mask = (mask > args.bound).float()
    else:
        mask = (mask > percentile(mask, args.perc)).float()

    logger.info("logging raw quality assignment...")

    for fid, (video_slices, mask_slice) in enumerate(
        zip(zip(*videos), mask.split(1))
    ):

        if fid % args.visualize_step_size == 0:

            image = T.ToPILImage()(video_slices[-1][0, :, :, :])
            visualize_heat_by_summarywriter(
                image,
                mask_slice.cpu().detach().float(),
                "raw_quality_assignment",
                writer,
                fid,
                args,
            )

    mask = dilate_binarize(mask, 0.5, args.conv_size, cuda=False)

    mask = postprocess_mask(mask)

    # logger.info("logging actual quality assignment...")

    # for fid, (video_slices, mask_slice) in enumerate(
    #     tqdm(zip(zip(*videos), mask.split(1)))
    # ):

    #     if fid % args.visualize_step_size == 0:

    #         image = T.ToPILImage()(video_slices[-1][0, :, :, :])
    #         visualize_heat_by_summarywriter(
    #             image,
    #             mask_slice.cpu().detach().float(),
    #             "quality_assignment",
    #             writer,
    #             fid,
    #             args,
    #         )

    # for i in range(len(mask)):

    #     for j in range(1, 31):

    #         if i + j >= len(mask):
    #             continue

    #         maski = mask[i : i + 1, :, :, :]
    #         maskj = mask[i + j : i + j + 1, :, :, :]
    #         iou = ((maski == 1) & (maskj == 1)).sum() / (
    #             (maski == 1) | (maskj == 1)
    #         ).sum()
    #         logger.info("Dist: %d, IoU: %.3f", j, iou.item())

    # if args.bound is not None:
    #     write_black_bkgd_video_smoothed_continuous(
    #         mask, args, args.qp, logger, writer=writer, tag="hq"
    #     )
    # else:
    #     perc_to_crf = {99: 0.5, 97: 1, 95: 1.5, 90: 2}
    #     write_black_bkgd_video_smoothed_continuous_crf(
    #         mask, args, perc_to_crf[args.perc], logger, writer=writer, tag="hq"
    #     )

    assert args.hq != -1

    if args.lq != -1:

        assert args.hq < args.lq
        assert "dual" in args.output

        orig_output = str(args.output)
        args.output = args.output + f".qp{args.hq}.mp4"
        write_black_bkgd_video_smoothed_continuous(
            mask, args, args.hq, logger, writer=writer, tag="hq"
        )

        mask = 1 - mask
        mask = dilate_binarize(mask, 0.5, 3, cuda=False)

        args.output = orig_output + f".base.mp4"
        write_black_bkgd_video_smoothed_continuous(
            mask, args, args.lq, logger, writer=writer, tag="lq"
        )

    else:

        assert "dual" not in args.output
        write_black_bkgd_video_smoothed_continuous(
            mask, args, args.hq, logger, writer=writer, tag="hq"
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
        "--app", type=str, help="The name of the model.", required=True,
    )

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
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=8
    )
    parser.add_argument(
        "--maskgen_file",
        type=str,
        help="The file that defines the neural network.",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="The path of pth file that stores the generator parameters.",
        required=True,
    )
    # parser.add_argument(
    #     "--upper_bound", type=float, help="The upper bound for the mask", required=True,
    # )
    # parser.add_argument(
    #     "--lower_bound", type=float, help="The lower bound for the mask", required=True,
    # )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--bound", type=float, help="The lower bound for the mask",
    )
    action.add_argument(
        "--perc", type=float, help="The percentage of modules to be encoded."
    )
    parser.add_argument(
        "--smooth_frames",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=30,
    )
    parser.add_argument(
        "--visualize_step_size",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=100,
    )
    parser.add_argument("--conv_size", type=int, default=1)
    parser.add_argument("--hq", type=int, default=-1)
    parser.add_argument("--lq", type=int, default=-1)

    # parser.add_argument('--mask', type=str,
    #                     help='The path of the ground truth video, for loss calculation purpose.', required=True)

    args = parser.parse_args()

    main(args)
