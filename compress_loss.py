"""
    Compress the video through gradient-based optimization.
"""

import argparse
import gc
import logging
from pathlib import Path
from pdb import set_trace

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
from utils.video_utils import get_qp_from_name, read_videos, write_video
from utils.visualize_utils import visualize_heat_by_summarywriter

sns.set()


def main(args):

    gc.enable()

    # initialize
    logger = logging.getLogger("meas")
    # logger.addHandler(logging.FileHandler("meas.log"))
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    videos = videos
    bws = [0, 1]
    qps = [get_qp_from_name(video_name) for video_name in video_names]

    # construct app
    app = DNN_Factory().get_model(args.app)

    # construct the mask
    mask_shape = [
        len(videos[-1]),
        1,
        720 // args.tile_size,
        1280 // args.tile_size,
    ]
    mask = torch.zeros(mask_shape).float()
    sum_grads = torch.zeros_like(mask)

    ground_truth_dict = read_results(args.ground_truth, app.name, logger)

    writer = SummaryWriter(f"runs/{args.app}/{args.output}")

    regions = [
        center_size(
            app.filter_result(i, args, gt=True)["instances"].pred_boxes.tensor
        )
        for i in ground_truth_dict.values()
    ]

    # set_trace()

    mask_obj = torch.clone(mask)
    for fid, mask_slice in enumerate(mask_obj.split(1)):
        mask_slice[:, :, :, :] = generate_mask_from_regions(
            mask_slice, regions[fid], 0, args.tile_size
        )

    for region in regions:
        region[:, 2:] = 0

    # first phase: enlargement
    for iteration in range(args.num_iterations + 1):

        tps = []

        logger.info(f"Processing application {app.name}")
        progress_bar = enlighten.get_manager().counter(
            total=len(videos[-1]), desc=f"{app.name}", unit="frames"
        )

        for fid, (video_slices, mask_slice) in enumerate(
            zip(zip(*videos), mask.split(1))
        ):

            progress_bar.update()

            # construct hybrid image, lq: normalized color, hq: hq image
            lq_image, hq_image = video_slices[0], video_slices[1]
            mean = torch.tensor([0.485, 0.456, 0.406])
            lq_image[:, :, :, :] = mean[None, :, None, None]
            mask_tile = tile_mask(mask_slice, args.tile_size)
            mix_image = lq_image * (1 - mask_tile) + hq_image * mask_tile

            # get result
            result = app.inference(mix_image.cuda(), detach=True)

            # get undetected bboxes
            index = app.get_undetected_ground_truth_index(
                result, ground_truth_dict[fid], args
            )

            if iteration == args.num_iterations:
                regions[fid][index, 2:] = 0
            else:
                regions[fid][index, 2:] += args.delta

            tps += [
                app.calc_accuracy(
                    {fid: result}, {fid: ground_truth_dict[fid]}, args
                )["f1"]
            ]

            # if tps[-1] < 0.9:
            #     regions[fid][:, 2:] += args.delta

            # logger.info("f1: %.3f", tps[-1])
            # logger.info(
            #     "Perc: %.3f", (mask_slice == 1).sum() / (mask_slice >= 0).sum()
            # )

            mask[fid : fid + 1, :, :, :] = generate_mask_from_regions(
                mask[fid : fid + 1, :, :, :], regions[fid], 0, args.tile_size
            )

            if (
                iteration == args.num_iterations
                and fid % args.visualize_step_size == 0
            ):

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

        logger.info("Average TP: %.3f", torch.tensor(tps).mean().item())

    # for fid in range(len(mask)):

    #     A = mask_obj[fid : fid + 1, :, :, :]
    #     B = mask[fid : fid + 1, :, :, :]

    #     logger.info(
    #         "IoU: %.3f",
    #         (((A == 1) & (B == 1)).sum() / ((A == 1) | (B == 1)).sum()).item(),
    #     )

    write_black_bkgd_video_smoothed_continuous(
        mask, args, args.qp, logger, writer=writer, tag="hq"
    )

    # return

    # mask = mask.cuda()
    # mask.requires_grad = True

    # for iteration in range(20):

    #     args.alpha = 1 / (iteration + 2)

    #     logger.info(f"Processing app {app.name}")
    #     progress_bar = enlighten.get_manager().counter(
    #         total=len(videos[-1]), desc=f"{app.name}", unit="frames"
    #     )

    #     losses = []
    #     f1s = []

    #     for fid, (video_slices, mask_slice) in enumerate(
    #         zip(zip(*videos), mask.split(1))
    #     ):

    #         progress_bar.update()

    #         lq_image, hq_image = video_slices[0], video_slices[1]
    #         lq_image = lq_image.cuda(non_blocking=True)
    #         hq_image = hq_image.cuda(non_blocking=True)

    #         # lq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_34/%05d.png' % (fid+offset2)))[None, :, :, :]

    #         # construct hybrid image
    #         mask_tile = tile_mask(mask_slice, args.tile_size)
    #         mix_image = lq_image * (1 - mask_tile) + hq_image * mask_tile
    #         loss = app.calc_accuracy_loss(
    #             mix_image, ground_truth_dict[fid], args
    #         )
    #         if isinstance(loss, torch.Tensor):
    #             loss.backward(retain_graph=True)
    #             losses.append(loss.item())
    #         else:
    #             losses.append(loss)

    #         # visualization
    #         if fid % args.visualize_step_size == 0:

    #             image = T.ToPILImage()(video_slices[-1][0, :, :, :])
    #             image = app.visualize(image, ground_truth_dict[fid], args)
    #             visualize_heat_by_summarywriter(
    #                 image,
    #                 mask_slice.cpu().detach().float(),
    #                 f"inferred_saliency_iter_{iteration}",
    #                 writer,
    #                 fid,
    #                 args,
    #             )

    #     mask.requires_grad = False
    #     grad = mask.grad
    #     grad = (grad - grad.min()) / (grad.max() - grad.min())
    #     sum_grads = sum_grads * 0.95 + mask.grad * 0.05
    #     for mask_slice, sum_slice in zip(
    #         mask.split(args.smooth_frames), sum_grads.split(args.smooth_frames)
    #     ):
    #         sum_slice_mean = sum_slice.mean(dim=0, keepdim=True)
    #         mask_slice[:, :, :, :] = torch.where(
    #             sum_slice_mean
    #             < percentile(sum_slice_mean, 100 - args.percentile),
    #             torch.ones_like(sum_slice_mean),
    #             torch.zeros_like(sum_slice_mean),
    #         )
    #     mask.grad.zero_()
    #     mask.requires_grad = True

    #     logger.info("The average loss is %.3f" % torch.tensor(losses).mean())

    #     logger.info("The average f1 is %.3f" % torch.tensor(f1s).mean())

    # mask.requires_grad = False
    # mask = mask.cpu()
    # write_black_bkgd_video_smoothed_continuous(
    #     mask, args, args.qp, logger, writer=writer, tag="hq"
    # )
    # # masked_video = generate_masked_video(mask, videos, bws, args)
    # # write_video(masked_video, args.output, logger)


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
        "--bound", type=float, help="The output name.", default=0.5
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
        "--smooth_frames",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=30,
    )
    parser.add_argument(
        "--delta",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=64,
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        help="Number of iterations needed",
        default=5,
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        help="The IoU threshold for calculating accuracy in object detection.",
        default=0.5,
    )
    parser.add_argument(
        "--percentile", type=float, help="The bound for the mask.", default=99,
    )
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=8
    )
    parser.add_argument(
        "--visualize_step_size", type=int, help="Visualization", default=100,
    )
    parser.add_argument(
        "--conv_size",
        type=int,
        help="Propose one single mask for smooth_frame many frames",
        default=1,
    )
    # parser.add_argument(
    #     "--upper_bound", type=float, help="The upper bound for the mask", required=True,
    # )
    # parser.add_argument(
    #     "--lower_bound", type=float, help="The lower bound for the mask", required=True,
    # )
    parser.add_argument(
        "--visualize",
        type=bool,
        help="Visualize the mask if True",
        default=False,
    )
    parser.add_argument("--qp", type=int, required=True)

    # parser.add_argument('--mask', type=str,
    #                     help='The path of the ground truth video, for loss calculation purpose.', required=True)

    args = parser.parse_args()

    main(args)
