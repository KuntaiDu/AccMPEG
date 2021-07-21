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
from torch.utils.tensorboard import SummaryWriter
from torchvision import io

#from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from dnn.CARN.interface import CARN
from dnn.dnn_factory import DNN_Factory
from utils.mask_utils import merge_black_bkgd_images
from utils.results_utils import write_results
from utils.video_utils import read_videos
from utils.video_utils import read_videos_pyav
from tqdm import tqdm

# a video is by default a 4-D Tensor [Time, Height, Width, Channel]


def main(args):

    logger = logging.getLogger("inference")
    handler = logging.NullHandler()
    logger.addHandler(handler)
    video = read_videos_pyav([args.input], logger)[0][0]

    # Construct image writer for visualization purpose
    writer = SummaryWriter(
        f"runs/{args.app}/{args.input}_{datetime.now().strftime(r'%d:%H:%M:%S')}"
    )

    app = DNN_Factory().get_model(args.app)
    if args.enable_cloudseg:
        super_resoluter = CARN()

    logger.info(f"Run %s on %s", app.name, args.input)
    inference_results = {}
    for fid, frame in enumerate(tqdm(video.decode(video=0), total=video.streams.video[0].frames)):
        video_slice = T.ToTensor()(frame.to_image()).unsqueeze(0)

        inference_results[fid] = app.inference(video_slice, detach=True)

        if fid % 100 == 0:
            image = T.ToPILImage()(
                F.interpolate(video_slice, (720, 1280))[0].cpu()
            )
            writer.add_image("decoded_image", T.ToTensor()(image), fid)

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
        "--enable_cloudseg",
        type=bool,
        help="Super-resolute the image before inference.",
        default=False,
    )

    args = parser.parse_args()

    main(args)
