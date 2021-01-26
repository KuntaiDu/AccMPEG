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

#from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from dnn.keypointrcnn_resnet50 import KeypointRCNN_ResNet50_FPN
from maskgen.vgg11 import FCN
#from maskgen.fcn_16_single_channel import FCN
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
    logger = logging.getLogger("blackgen")
    logger.addHandler(logging.FileHandler("blackgen.log"))
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    videos = videos
    bws = [0, 1]
    qps = [get_qp_from_name(video_name) for video_name in video_names]

    # construct applications
    application = KeypointRCNN_ResNet50_FPN() #FasterRCNN_ResNet50_FPN()

    mask_generator = FCN()
    mask_generator.load(args.path)
    mask_generator.eval().cuda()

    # construct the mask
    mask_shape = [len(videos[-1]), 1, 720 // args.tile_size, 1280 // args.tile_size]
    mask = torch.ones(mask_shape).float()

    ground_truth_dict = read_results(
        args.ground_truth, "KeypointRCNN_ResNet50_FPN", logger
    )
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
                with Timer("maskgen", logger):
                    mask_gen = mask_generator(hq_image)
                # losses.append(get_loss(mask_gen, ground_truth_mask[fid]))
                mask_gen = mask_gen.softmax(dim=1)[:, 1:2, :, :]
                mask_lb = dilate_binarize(mask_gen, args.bound, args.conv_size)
                # mask_ub = dilate_binarize(mask_gen, args.upper_bound, args.conv_size)
                mask_slice[:, :, :, :] = mask_lb
                # mask_slice[:, :, :, :] = torch.where(mask_gen > 0.5, torch.ones_like(mask_gen), torch.zeros_like(mask_gen))

            # mask_slice[:, :, :, :] = ground_truth_mask[fid + offset2].float()

            # lq_image[:, :, :, :] = background
            # # calculate the loss, to see the generalization error
            # with torch.no_grad():
            #     mask_slice = tile_mask(mask_slice, args.tile_size)
            #     masked_image = generate_masked_image(
            #         mask_slice, video_slices, bws)

            #     video_results = application.inference(
            #         masked_image.cuda(), True)[0]
            #     f1s.append(application.calc_accuracy({
            #         fid: video_results
            #     }, {
            #         fid: ground_truth_dict[fid]
            #     }, args)['f1'])

            # import pdb; pdb.set_trace()
            # loss, _ = application.calc_loss(masked_image.cuda(),
            #                                 application.inference(video_slices[-1].cuda(), detach=True)[0], args)
            # total_loss.append(loss.item())

            # visualization
            if args.visualize and (fid % 100 == 0):
                heat = tile_mask(mask_gen, args.tile_size)[0, 0, :, :]
                fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
                ax = sns.heatmap(
                    heat.cpu().detach().numpy(),
                    zorder=3,
                    alpha=0.5,
                    ax=ax,
                    xticklabels=False,
                    yticklabels=False,
                )  # 1.3s
                with torch.no_grad():
                    inf = application.inference(hq_image, detach=True)[0]
                image = T.ToPILImage()(video_slices[-1][0, :, :, :])
                image = application.plot_results_on(inf, image, (255, 255, 255), args)
                # image = application.plot_results_on(video_results, image, (0, 255, 255), args)
                ax.imshow(image, zorder=3, alpha=0.5)
                ax.tick_params(left=False, bottom=False)
                Path(f"heat/{args.output}/").mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    f"heat/{args.output}/{fid}.png", bbox_inches="tight"
                )  # 4.6s ==> 1.1s

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
    parser.add_argument(
        "-g",
        "--ground_truth",
        help="The video file names. The largest video file will be the ground truth.",
        type=str,
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
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=8
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
    parser.add_argument(
        "--bound", type=float, help="The lower bound for the mask", required=True,
    )
    parser.add_argument(
        "--visualize", type=bool, help="Visualize the mask if True", default=False,
    )
    parser.add_argument(
        "--smooth_frames",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=250,
    )
    parser.add_argument("--conv_size", type=int, default=1)
    parser.add_argument("--force_qp", type=int, default=-1)

    # parser.add_argument('--mask', type=str,
    #                     help='The path of the ground truth video, for loss calculation purpose.', required=True)

    args = parser.parse_args()

    main(args)
