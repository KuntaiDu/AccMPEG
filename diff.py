import argparse
import logging
from pathlib import Path

import coloredlogs
import enlighten
import torch
import torchvision.transforms as T
from kornia.color.yuv import rgb_to_yuv

from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from utils.video_utils import read_video


def main(args):

    logger = logging.getLogger("diff")
    torch.set_default_tensor_type(torch.FloatTensor)

    x = read_video(args.inputs[0], logger, True, False)
    y = read_video(args.inputs[1], logger, True, False)

    Path(args.output[0]).mkdir(parents=True, exist_ok=True)

    progress_bar = enlighten.get_manager().counter(
        total=len(x), desc=f"Getting diff", unit="frames",
    )

    application = FasterRCNN_ResNet50_FPN()
    application.cuda()

    for fid, (imx, imy) in enumerate(zip(x, y)):

        progress_bar.update()

        result_x = application.inference(imx.cuda(), detach=True)[0]
        _, _, x_boxes, _ = application.filter_results(
            result_x, args.confidence_threshold
        )
        result_y = application.inference(imy.cuda(), detach=True)[0]
        _, _, y_boxes, _ = application.filter_results(
            result_y, args.confidence_threshold
        )

        image = T.ToPILImage()(imx[0, :, :, :])
        image = application.plot_results_on(result_x, image, "Azure", args, y_boxes)
        image = application.plot_results_on(result_y, image, "SteelBlue", args, x_boxes)

        image.save(args.output[0] + "/%010d.png" % fid)


if __name__ == "__main__":

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--inputs", nargs=2, type=str, help="The video file names.", required=True
    )

    parser.add_argument(
        "-o", "--output", nargs=1, type=str, help="The output file name", required=True
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

    args = parser.parse_args()

    main(args)
