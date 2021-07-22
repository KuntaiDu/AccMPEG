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
from torch.utils.tensorboard import SummaryWriter
from torchvision import io

from dnn.dnn_factory import DNN_Factory
from maskgen.vgg11 import FCN
from utils.bbox_utils import center_size
from utils.loss_utils import focal_loss as get_loss
from utils.mask_utils import *
from utils.results_utils import read_ground_truth, read_results
from utils.timer import Timer
from utils.video_utils import get_qp_from_name, read_videos, read_videos_pyav, write_video
from utils.visualize_utils import visualize_heat_by_summarywriter
from tqdm import tqdm

# added this summer
import av

sns.set()


def main(args):

    gc.enable()

    # initialize
    logger = logging.getLogger("maskgen")
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    # videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    video = read_videos_pyav([args.input], logger)[0][0]

    # construct applications
    # app = DNN_Factory().get_model(args.app)

    mask_generator = FCN()
    mask_generator.load(args.path)
    mask_generator.eval().cuda()

    # num_frames = len([f for f in videos[0].decode()])
    num_frames = video.streams.video[0].frames
    # construct the mask
    mask_shape = [
        # len(videos[-1]),
        num_frames,
        1,
        720 // args.tile_size,
        1280 // args.tile_size,
    ]
    mask = torch.ones(mask_shape).float()

    # construct the writer for writing the result
    writer = SummaryWriter(f"runs/{args.output}")


    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)
    logger.info("Start mask generation...")

    for fid, (hq_frame, mask_slice) in enumerate(
        tqdm(zip(video.decode(video=0), mask.split(1)), total=num_frames)
    ):
        hq_image = T.ToTensor()(hq_frame.to_image()).unsqueeze(0)

        # construct hybrid image
        with torch.no_grad():
            hq_image = hq_image.cuda()
            mask_gen = mask_generator(hq_image)
            mask_gen = mask_gen.softmax(dim=1)[:, 1:2, :, :]
            mask_slice[:, :, :, :] = mask_gen

        # visualization
        if fid % args.visualize_step_size == 0:

            image = T.ToPILImage()(hq_image.cpu()[0, :, :, :])

            visualize_heat_by_summarywriter(
                image,
                mask_slice.cpu().detach().float(),
                "inferred_saliency",
                writer,
                fid,
                args,
            )


    # qizheng: instead, store the mask information in a separate file
    with open(f"{args.output}.rawmask", "wb") as f:
        pickle.dump(mask, f)

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
        help="The video file name that generates the mask.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The video file name of the final compressed video.",
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=1)
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
        "--visualize_step_size",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=100,
    )

    args = parser.parse_args()

    main(args)
