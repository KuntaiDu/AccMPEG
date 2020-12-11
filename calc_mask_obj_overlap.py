"""
    Compress the video through gradient-based optimization.
"""

import argparse
import gc
import logging
from pathlib import Path

import coloredlogs
import enlighten
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image, ImageDraw
from torchvision import io

from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from maskgen.fcn_16_single_channel import FCN
from utils.loss_utils import focal_loss as get_loss
from utils.mask_utils import *
from utils.results_utils import read_ground_truth, read_results
from utils.video_utils import get_qp_from_name, read_videos, write_video

sns.set()


def main(args):

    gc.enable()

    # initialize
    logger = logging.getLogger("calc")
    logger.addHandler(logging.FileHandler("calc.log"))
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    videos = videos
    bws = [0, 1]
    qps = [get_qp_from_name(video_name) for video_name in video_names]

    # construct applications
    application = FasterRCNN_ResNet50_FPN()

    mask_generator = FCN()
    mask_generator.load(args.path)
    mask_generator.train().cuda()

    # construct the mask
    mask_shape = [len(videos[-1]), 1, 720 // args.tile_size, 1280 // args.tile_size]
    mask = torch.ones(mask_shape).float()

    ground_truth_dict = read_results(args.inputs[-1], "FasterRCNN_ResNet50_FPN", logger)
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

    losses = []
    f1s = []
    fn1s = []
    fn2s = []
    areas = []

    for temp in range(1):

        logger.info(f"Processing application {application.name}")
        progress_bar = enlighten.get_manager().counter(
            total=len(videos[-1]), desc=f"{application.name}", unit="frames"
        )

        application.cuda()

        for fid, (video_slices, mask_slice) in enumerate(
            zip(zip(*videos), mask.split(1))
        ):

            progress_bar.update()

            _, hq_image = video_slices[0], video_slices[1]
            # lq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_34/%05d.png' % (fid+offset2)))[None, :, :, :]

            visualize_flag = False
            if args.visualize and fid % 50 == 0:
                visualize_flag = True

            # construct hybrid image
            with torch.no_grad():
                # mask_gen = mask_generator(
                #     torch.cat([hq_image, hq_image - lq_image], dim=1).cuda()
                # )
                mask_gen = mask_generator(hq_image.cuda())
                # losses.append(get_loss(mask_gen, ground_truth_mask[fid]))
                mask_gen = mask_gen.softmax(dim=1)[:, 1:2, :, :]
                mask_slice[:, :, :, :] = dilate_binarize(
                    mask_gen, args.bound, args.conv_size
                ).cpu()

            mask_slice_tiled = tile_mask(mask_slice, args.tile_size)

            tp, fn1, fn2 = 0, 0, 0
            _, _, boxes, _ = application.filter_results(
                ground_truth_dict[fid], args.confidence_threshold
            )

            image, draw = None, None
            if visualize_flag:
                image = T.ToPILImage()(hq_image[0, :, :, :])
                draw = ImageDraw.Draw(image)

            for box in boxes:

                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                assert y2 <= 720

                box_mask = torch.zeros_like(mask_slice_tiled)
                box_mask[:, :, y1:y2, x1:x2] = 1
                overlap = box_mask * mask_slice_tiled
                if torch.equal(box_mask * mask_slice_tiled, box_mask):
                    tp += 1
                    if visualize_flag:
                        draw.rectangle([x1, y1, x2, y2], width=6, outline="white")
                elif overlap.sum() > 0:
                    fn1 += 1
                    if visualize_flag:
                        draw.rectangle([x1, y1, x2, y2], width=6, outline="steelblue")
                else:
                    fn2 += 1
                    if visualize_flag:
                        draw.rectangle([x1, y1, x2, y2], width=6, outline="red")

            if visualize_flag:
                Path("visualize/" + args.output).mkdir(exist_ok=True)
                heat = mask_slice_tiled[0, 0, :, :]
                plt.clf()
                ax = sns.heatmap(heat.cpu().detach().numpy(), zorder=3, alpha=0.5)
                ax.imshow(image, zorder=3, alpha=0.5)
                plt.savefig(
                    "visualize/" + args.output + "/%010d.png" % fid, bbox_inches="tight"
                )

            f1s.append(tp * 1.0 / (tp + fn1 + fn2))
            fn1s.append(fn1)
            fn2s.append(fn2)

            areas.append(mask_slice.mean())

        application.cpu()

    mask.requires_grad = False
    mask = binarize_mask(mask, bws)
    qps = [min(qps)]
    if args.force_qp:
        qps = [args.force_qp]
    write_black_bkgd_video_smoothed(mask, args, qps, bws, logger, 5)

    with open("stats_overlap", "a") as f:
        f.write(
            yaml.dump(
                [
                    {
                        "video_name": args.inputs[-1],
                        "acc": torch.Tensor(f1s).mean().item(),
                        "f1": os.path.getsize(args.output),
                        "area": torch.Tensor(areas).mean().item(),
                        "fn1": torch.Tensor(fn1s).float().mean().item(),
                        "fn2": torch.Tensor(fn2s).float().mean().item(),
                        "bound": args.bound,
                    }
                ]
            )
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
        "--confidence_threshold",
        type=float,
        help="The confidence score threshold for calculating accuracy.",
        default=0.3,
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
    parser.add_argument(
        "--tile_percentage",
        type=float,
        help="How many percentage of tiles will remain",
        default=1,
    )
    parser.add_argument(
        "--bound", type=float, help="The upper bound for the mask", required=True,
    )
    parser.add_argument(
        "--visualize", type=bool, help="Visualize the mask if True", default=False,
    )
    parser.add_argument("--conv_size", type=int, default=1)
    parser.add_argument("--force_qp", type=int, default=-1)

    # parser.add_argument('--mask', type=str,
    #                     help='The path of the ground truth video, for loss calculation purpose.', required=True)

    args = parser.parse_args()

    main(args)
