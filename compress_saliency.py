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
from torchvision import io

from dnn.dnn_factory import DNN_Factory
from maskgen.fcn_16_single_channel import FCN
from utils.bbox_utils import center_size
from utils.loss_utils import focal_loss as get_loss
from utils.mask_utils import *
from utils.results_utils import read_ground_truth, read_results
from utils.video_utils import get_qp_from_name, read_videos, write_video
from utils.visualize_utils import visualize_heat

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

    # construct apps
    app = DNN_Factory().get_model(args.app)

    # construct the mask
    mask_shape = [
        len(videos[-1]),
        1,
        720 // args.tile_size,
        1280 // args.tile_size,
    ]
    mask = torch.ones(mask_shape).float()
    # mask2 = torch.ones(mask_shape).float()

    # ###
    # ###
    # ###
    # mask_generator = FCN()
    # mask_generator.load(
    #     "maskgen_pths/COCO_full_normalizedsaliency_vgg11_crossthresh.pth.best"
    # )
    # mask_generator.eval().cuda()
    # ###
    # ###
    # ###

    ground_truth_dict = read_results(args.ground_truth, app.name, logger)
    # logger.info('Reading ground truth mask')
    # with open(args.mask + '.mask', 'rb') as f:
    #     ground_truth_mask = pickle.load(f)
    # ground_truth_mask = ground_truth_mask[sorted(ground_truth_mask.keys())[1]]
    # ground_truth_mask = ground_truth_mask.split(1)

    # binarized_mask = mask.clone().detach()
    # binarize_mask(binarized_mask, bws)
    # if iteration > 3 * (args.num_iterations // 4):
    #     (args.binarize_weight * torch.tensor(iteration*1.0) * (binarized_mask - mask).abs().pow(2).mean()).backward()

    for temp in range(1):

        logger.info(f"Processing app {app.name}")
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

            # construct hybrid image
            lq_image = (
                torch.ones_like(hq_image)
                * torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
            )
            lq_image.requires_grad = True
            gt_result = ground_truth_dict[fid]
            loss = app.calc_loss(lq_image, gt_result, args)

            loss.backward()
            # mask_grad = hq_image.grad.norm(dim=1, p=2, keepdim=True)
            with torch.no_grad():
                mask_grad = lq_image.grad.cuda()
                mask_grad = mask_grad ** 2
                mask_grad = mask_grad.sum(dim=1, keepdim=True)
                mask_grad = F.conv2d(
                    mask_grad,
                    torch.ones([1, 1, args.tile_size, args.tile_size]).cuda(),
                    stride=args.tile_size,
                ).cpu()

                # diff = (hq_image - lq_image).cuda()
                # diff = diff ** 2
                # diff = diff.sum(dim=1, keepdim=True)
                # diff = F.conv2d(
                #     diff,
                #     torch.ones([1, 1, args.tile_size, args.tile_size]).cuda(),
                #     stride=args.tile_size,
                # ).cpu()

                # mask_grad = mask_grad * diff
                mask_grad = mask_grad.sqrt()
                mask_grad = (mask_grad - mask_grad.min()) / (
                    mask_grad.max() - mask_grad.min()
                )
                mask_slice[:, :, :, :] = mask_grad
                # mask_slice[:, :, :, :] = dilate_binarize(
            #     mask_grad, args.bound, args.conv_size, True,
            # ).cpu()
            # mask_gen = mask_generator(
            #     torch.cat([hq_image, hq_image - lq_image], dim=1).cuda()
            # )
            # mask_gen = mask_generator(hq_image.cuda())
            # # losses.append(get_loss(mask_gen, ground_truth_mask[fid]))
            # mask_gen = mask_gen.softmax(dim=1)[:, 1:2, :, :]
            # mask_lb = dilate_binarize(mask_gen, args.lower_bound, args.conv_size)
            # mask_ub = dilate_binarize(mask_gen, args.upper_bound, args.conv_size)
            # mask_slice[:, :, :, :] = mask_lb - mask_ub
            # mask_slice[:, :, :, :] = torch.where(mask_gen > 0.5, torch.ones_like(mask_gen), torch.zeros_like(mask_gen))
            # mask_slice[:, :, :, :] = ground_truth_mask[fid + offset2].float()

            # lq_image[:, :, :, :] = background
            # # calculate the loss, to see the generalization error
            # with torch.no_grad():
            #     mask_slice = tile_mask(mask_slice, args.tile_size)
            #     masked_image = generate_masked_image(
            #         mask_slice, video_slices, bws)

            #     video_results = app.inference(
            #         masked_image.cuda(), True)[0]
            #     f1s.append(app.calc_accuracy({
            #         fid: video_results
            #     }, {
            #         fid: ground_truth_dict[fid]
            #     }, args)['f1'])

            # import pdb; pdb.set_trace()
            # loss, _ = app.calc_loss(masked_image.cuda(),
            #                                 app.inference(video_slices[-1].cuda(), detach=True)[0], args)
            # total_loss.append(loss.item())

            # visualize by default.
            if fid % 50 == 0:
                heat = mask_slice.cpu().detach()
                image = T.ToPILImage()(video_slices[-1][0, :, :, :])
                image = app.visualize(
                    image, app.filter_result(gt_result, args), args
                )
                visualize_heat(
                    image,
                    heat,
                    f"visualize/{args.output}/{app.name}/saliency/%010d.png"
                    % fid,
                    args,
                )
                # hq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_24/%05d.png' % (fid+offset2)))[None, :, :, :].cuda()
                # with torch.no_grad():
                #     inf = app.inference(hq_image, detach=True)[0]

                # image = app.plot_results_on(video_results, image, (0, 255, 255), args)

    mask.requires_grad = False
    mask = dilate_binarize(mask, args.bound, args.conv_size, cuda=False)
    write_black_bkgd_video_smoothed_continuous(
        mask, args, args.force_qp, logger
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
        default=0.5,
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        help="The IoU threshold for calculating accuracy in object detection.",
        default=0.5,
    )
    parser.add_argument(
        "--bound", type=float, help="The bound for the mask.", default=0.5,
    )
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=8
    )
    parser.add_argument(
        "--smooth_frames",
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
    parser.add_argument("--conv_size", type=int, required=True)
    parser.add_argument("--qp", type=int, default=-1)

    # parser.add_argument('--mask', type=str,
    #                     help='The path of the ground truth video, for loss calculation purpose.', required=True)

    args = parser.parse_args()

    main(args)
