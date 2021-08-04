"""
    Compress the video through gradient-based optimization.
"""

import argparse
import gc
import logging
from pathlib import Path
from pdb import set_trace

import coloredlogs
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision import io

# from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
# from dnn.keypointrcnn_resnet50 import KeypointRCNN_ResNet50_FPN
from dnn.dnn_factory import DNN_Factory
from maskgen.fcn_16_single_channel import FCN
from utils.bbox_utils import center_size
from utils.loss_utils import focal_loss as get_loss
from utils.mask_utils import *
from utils.results_utils import read_ground_truth, read_results
from utils.video_utils import get_qp_from_name, read_video_pyav, write_video
from utils.visualize_utils import visualize_heat

from tqdm import tqdm

sns.set()


def main(args):

    gc.enable()

    # initialize
    logger = logging.getLogger("blackgen")
    logger.addHandler(logging.FileHandler("blackgen.log"))
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    video = read_video_pyav(args.hq, logger)

    # construct apps
    app = DNN_Factory().get_model(args.app)

    # construct the mask
    mask_shape = [
        video.streams.video[0].frames,
        1,
        720 // args.tile_size,
        1280 // args.tile_size,
    ]
    mask = torch.ones(mask_shape).float()

    ground_truth_dict = read_results(args.ground_truth, app.name, logger)


    logger.info(f"Processing app {app.name}")

    losses = []
    f1s = []

    for fid, (image, mask_slice) in enumerate(
        zip(video.decode(video=0), tqdm(mask.split(1)))
    ):


        # lq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_34/%05d.png' % (fid+offset2)))[None, :, :, :]

        raw_image = image.to_image()

        
        image = T.ToTensor()(image.to_image()).unsqueeze(0).cuda()
        image.requires_grad = True

        gt_result = ground_truth_dict[fid]
        loss = app.calc_loss(image, gt_result, args)

        loss.backward()
        # mask_grad = hq_image.grad.norm(dim=1, p=2, keepdim=True)
        with torch.no_grad():
            mask_grad = image.grad.cuda()
            mask_grad = mask_grad ** 2
            mask_grad = mask_grad.sum(dim=1, keepdim=True)
            mask_grad = F.conv2d(
                mask_grad,
                torch.ones([1, 1, args.tile_size, args.tile_size]).cuda(),
                stride=args.tile_size,
            ).cpu()

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
        if fid % 1 == 0:
            heat = mask_slice.cpu().detach()
            raw_image = app.visualize(
                raw_image, app.filter_result(gt_result, args), args
            )
            visualize_heat(
                raw_image,
                heat,
                f"visualize/{args.output}/{app.name}/saliency/%010d.png"
                % fid,
                args,
            )
            # hq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_24/%05d.png' % (fid+offset2)))[None, :, :, :].cuda()
            # with torch.no_grad():
            #     inf = app.inference(hq_image, detach=True)[0]

            # image = app.plot_results_on(video_results, image, (0, 255, 255), args)

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
        "--hq",
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
