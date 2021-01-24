import argparse
import logging
import pickle
from pathlib import Path
from pdb import set_trace

import coloredlogs
import enlighten
import torch
import torchvision.transforms as T
from torchvision import io

from dnn.CARN.interface import CARN
from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from dnn.fcn_resnet50 import FCN_ResNet50
from utils.results_utils import write_results
from utils.video_utils import read_videos

# a video is by default a 4-D Tensor [Time, Height, Width, Channel]


def main(args):

    logger = logging.getLogger("inference")
    handler = logging.NullHandler()
    logger.addHandler(handler)

    videos, _, video_names = read_videos(
        args.inputs, logger, normalize=False, from_source=False
    )
    # from utils.video_utils import write_video
    # write_video(videos[0], 'download.mp4', logger)

    application_bundle = [FCN_ResNet50()]
    if args.enable_cloudseg:
        super_resoluter = CARN()

    for application in application_bundle:

        # put the application on GPU
        application.cuda()

        for vid, video in enumerate(videos):

            video_name = video_names[vid]
            logger.info(f"Run {application.name} on {video_name}")
            progress_bar = enlighten.get_manager().counter(
                total=len(video),
                desc=f"{application.name}: {video_name}",
                unit="frames",
            )
            inference_results = {}

            for fid, video_slice in enumerate(video):
                progress_bar.update()

                # transforms = T.Compose(
                #     [
                #         T.ToPILImage(),
                #         T.ColorJitter(0.05, 0.05, 0.05, 0.05),
                #         T.ToTensor(),
                #     ]
                # )

                video_slice = video_slice.cuda()

                if args.enable_cloudseg:
                    video_slice = super_resoluter(video_slice)

                # video_slice = transforms(video_slice[0])[None, :, :, :]
                # video_slice = video_slice + torch.randn_like(video_slice) * 0.05
                inference_results[fid] = application.inference(
                    video_slice, detach=True
                )[0]

                if fid % 10 == 0:
                    folder = Path("inference/" + video_name)
                    folder.mkdir(exist_ok=True, parents=True)
                    image = T.ToPILImage()(video_slice[0].cpu())
                    image = application.plot_results_on(
                        inference_results[fid], image, "Azure", args
                    )
                    image.save(folder / ("%010d.png" % fid))

            write_results(video_name, application.name, inference_results, logger)


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
        type=str,
        help="The video file names to obtain inference results.",
        required=True,
        nargs="+",
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
        "--enable_cloudseg",
        type=bool,
        help="Super-resolute the image before inference.",
        default=False,
    )

    args = parser.parse_args()

    main(args)
