import argparse
import logging
from pathlib import Path

import coloredlogs
import enlighten
import torch
import torchvision.transforms as T
from kornia.color.yuv import rgb_to_yuv

from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from utils.results_utils import read_results
from utils.video_utils import read_video


def main(args):

    logger = logging.getLogger("diff")
    torch.set_default_tensor_type(torch.FloatTensor)

    x = read_video(args.inputs[0], logger, True, False)

    Path(args.output[0]).mkdir(parents=True, exist_ok=True)

    progress_bar = enlighten.get_manager().counter(
        total=len(x), desc=f"Getting diff", unit="frames",
    )

    application = FasterRCNN_ResNet50_FPN()
    application.cuda()

    result_x = read_results(args.inputs[0], application.name, logger)
    result_y = read_results(args.inputs[1], application.name, logger)

    for fid, imx in enumerate(x):

        progress_bar.update()

        if fid % 100 != 0:
            continue

        _, _, x_boxes, _ = application.filter_results(
            result_x[fid], args.confidence_threshold
        )
        _, _, y_boxes, _ = application.filter_results(
            result_y[fid], args.confidence_threshold
        )

        image = T.ToPILImage()(imx[0, :, :, :])
        image = application.plot_results_on(
            result_x[fid], image, "Azure", args, y_boxes
        )
        image = application.plot_results_on(
            result_y[fid], image, "SteelBlue", args, x_boxes
        )

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
        default=0.5,
    )

    parser.add_argument(
        "--iou_threshold",
        type=float,
        help="The IoU threshold for calculating accuracy in object detection.",
        default=0.5,
    )

    args = parser.parse_args()

    main(args)
