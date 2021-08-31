import argparse
import glob
import logging
import pickle
from datetime import datetime
from pathlib import Path
from pdb import set_trace

import coloredlogs
import enlighten
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import io

from dnn.CARN.interface import CARN
from dnn.dnn_factory import DNN_Factory
from utils.mask_utils import merge_black_bkgd_images
from utils.results_utils import read_results, write_results
from utils.timer import Timer
from utils.video_utils import read_videos

# from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN


# a video is by default a 4-D Tensor [Time, Height, Width, Channel]


def main(args):

    logger = logging.getLogger("inference")
    handler = logging.NullHandler()
    logger.addHandler(handler)

    if "dual" not in args.input:
        assert args.from_source == False
        videos, _, _ = read_videos(
            [args.input], logger, normalize=False, from_source=args.from_source
        )
    else:
        # set_trace()
        ext = args.input.split(".")[-1]
        # set_trace()
        assert len(glob.glob(args.input + f"*.{ext}")) == 2

        videos, _, _ = read_videos(
            sorted(glob.glob(args.input + f"*.{ext}")),
            logger,
            normalize=False,
            from_source=args.from_source,
        )

    # Construct image writer for visualization purpose
    writer = SummaryWriter(f"runs/{args.app}/{args.input}")

    app = DNN_Factory().get_model(args.app)
    if args.enable_cloudseg:
        super_resoluter = CARN()

    # load ground truth for visualization purpose
    if args.ground_truth == "":
        ground_truth_dict = None
    else:
        ground_truth_dict = read_results(args.ground_truth, app.name, logger)

    logger.info(f"Run %s on %s", app.name, args.input)
    progress_bar = enlighten.get_manager().counter(
        total=len(videos[0]), desc=f"{app.name}: {args.input}", unit="frames",
    )
    inference_results = {}

    for fid, video_slice in enumerate(zip(*videos)):

        if "dual" in args.input:
            hq_video_slice = video_slice[1]
            video_slice = merge_black_bkgd_images(video_slice)
        else:
            video_slice = video_slice[0]
            hq_video_slice = video_slice
        progress_bar.update()

        # video_slice = video_slice.cuda()

        if args.enable_cloudseg:
            assert (
                "dual" not in args.input
            ), "Dual does not work well with cloudseg."
            video_slice = super_resoluter(video_slice.cuda())

        # video_slice = transforms(video_slice[0])[None, :, :, :]
        # video_slice = video_slice + torch.randn_like(video_slice) * 0.05
        # with Timer("inference", logger):
        inference_results[fid] = app.inference(video_slice, detach=True)

        if fid % args.visualize_step_size == 0:
            image = T.ToPILImage()(
                F.interpolate(video_slice, (720, 1280))[0].cpu()
            )

            # image2 = Image.open(
            #     "DAVIS/videos/DAVIS_1_qp_30.mp4.pngs/%010d.png" % fid
            # )
            writer.add_image("decoded_image", T.ToTensor()(image), fid)
            # writer.add_image(
            #     "diff", (T.ToTensor()(image) - T.ToTensor()(image2)) + 0.3, fid
            # )
            image = app.visualize(image, inference_results[fid], args)
            writer.add_image("inference_result", T.ToTensor()(image), fid)

            if ground_truth_dict is not None:
                hq_image = T.ToPILImage()(
                    F.interpolate(hq_video_slice, (720, 1280))[0].cpu()
                )
                (
                    gt_index,
                    result_index,
                    gt,
                    result,
                ) = app.get_undetected_ground_truth_index(
                    inference_results[fid], ground_truth_dict[fid], args
                )

                # Visualize false negatives
                writer.add_image(
                    "FN",
                    T.ToTensor()(
                        app.visualize(
                            hq_image, {"instances": gt[gt_index]}, args
                        )
                    ),
                    fid,
                )

                writer.add_image(
                    "FP",
                    T.ToTensor()(
                        app.visualize(
                            hq_image, {"instances": result[result_index]}, args
                        )
                    ),
                    fid,
                )

    write_results(args.input, app.name, inference_results, logger)


if __name__ == "__main__":

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="The video file names to obtain inference results.",
        required=True,
    )
    parser.add_argument(
        "--app", type=str, help="The name of the model.", required=True,
    )
    parser.add_argument(
        "--visualize_step_size",
        type=int,
        help="The name of the model.",
        default=100,
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
        "--enable_cloudseg",
        type=bool,
        help="Super-resolute the image before inference.",
        default=False,
    )
    parser.add_argument(
        "--from_source", type=bool, help="No reencoding?", default=False,
    )
    parser.add_argument(
        "--ground_truth",
        "-g",
        type=str,
        help="The ground truth (for visualization purpose)",
        default="",
    )

    args = parser.parse_args()

    main(args)
