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
from torchvision import io

from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from maskgen.vgg11 import FCN
from utils.bbox_utils import center_size
from utils.loss_utils import focal_loss as get_loss
from utils.mask_utils import *
from utils.results_utils import read_ground_truth, read_results
from utils.timer import Timer
from utils.video_utils import get_qp_from_name, read_videos, write_video

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
    application = FasterRCNN_ResNet50_FPN()

    # construct the mask
    mask_shape = [len(videos[-1]), 1, 720 // args.tile_size, 1280 // args.tile_size]
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

    for temp in range(1):

        logger.info(f"Processing application {application.name}")
        progress_bar = enlighten.get_manager().counter(
            total=len(videos[-1]), desc=f"{application.name}", unit="frames"
        )

        application.cuda()

        losses = []
        f1s = []

        for fid, (video_slices, mask_slice) in enumerate(
            zip(zip(*videos), mask.split(1))
        ):

            progress_bar.update()

            lq_image, hq_image = video_slices[0], video_slices[1]
            # lq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_34/%05d.png' % (fid+offset2)))[None, :, :, :]

            hq_image = hq_image.cuda()
            regions = application.region_proposal(hq_image)[0]
            regions = center_size(regions).cpu()
            sizes = regions[:, 2] * regions[:, 3]
            regions[sizes > (720 * 1280 * 0.06), :] = 0

            maskB = generate_mask_from_regions(
                mask_slice.clone(), regions, 0, args.tile_size
            )
            mask_delta = maskB
            mask_delta[mask_delta < 0] = 0
            mask_slice[:, :, :, :] = mask_delta

            # visualization
            if args.visualize and (fid % 100 == 0):
                heat = tile_mask(mask_slice, args.tile_size)[0, 0, :, :]
                plt.clf()
                ax = sns.heatmap(heat.cpu().detach().numpy(), zorder=3, alpha=0.5)
                # hq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_24/%05d.png' % (fid+offset2)))[None, :, :, :].cuda()
                # with torch.no_grad():
                #     inf = application.inference(hq_image, detach=True)[0]
                image = T.ToPILImage()(video_slices[-1][0, :, :, :])
                # image = application.plot_results_on(inf, image, (255, 255, 255), args)
                # image = application.plot_results_on(video_results, image, (0, 255, 255), args)
                ax.imshow(image, zorder=3, alpha=0.5)
                Path(f"visualize/{args.output}/").mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    f"visualize/{args.output}/{fid}_attn.png", bbox_inches="tight"
                )

            # plt.clf()
            # sns.distplot(heat.flatten().detach().numpy())
            # plt.savefig(
            #     f"visualize/{args.output}/{fid}_dist.png", bbox_inches="tight"
            # )

        logger.info("In video %s", args.output)
        logger.info("The average loss is %.3f" % torch.tensor(losses).mean())

        with open("temp.txt", "w") as f:
            f.write(f"{torch.tensor(f1s).mean()}")
        logger.info("The average f1 is %.3f" % torch.tensor(f1s).mean())

        application.cpu()

    mask.requires_grad = False
    qps = [min(qps)]
    if args.force_qp != -1:
        qps = [args.force_qp]
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "-s", "--source", type=str, help="The original video source.", required=True
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
    parser.add_argument(
        "--visualize", type=bool, help="Visualize the mask if True", default=False,
    )
    parser.add_argument(
        "--smooth_frames",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=1,
    )
    parser.add_argument("--conv_size", type=int, default=1)
    parser.add_argument("--force_qp", type=int, default=-1)
    parser.add_argument("--bound", type=float, default=0.5)

    # parser.add_argument('--mask', type=str,
    #                     help='The path of the ground truth video, for loss calculation purpose.', required=True)

    args = parser.parse_args()

    main(args)
