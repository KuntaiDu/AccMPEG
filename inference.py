
import torch
from torchvision import io
import argparse
import coloredlogs
import logging
import enlighten
import pickle
from pathlib import Path

from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from utils.video_utils import read_videos
from utils.results_utils import write_results

# a video is by default a 4-D Tensor [Time, Height, Width, Channel]

def main(args):

    logger = logging.getLogger('inference')
    handler = logging.NullHandler()
    logger.addHandler(handler)

    videos, _, video_names = read_videos(args.inputs, logger, normalize=False)
    # from utils.video_utils import write_video
    # write_video(videos[0], 'download.mp4', logger)

    application_bundle = [FasterRCNN_ResNet50_FPN()]

    for application in application_bundle:

        # put the application on GPU
        application.cuda()

        for vid, video in enumerate(videos):

            video_name = video_names[vid]
            logger.info(f'Run {application.name} on {video_name}')
            progress_bar = enlighten.get_manager().counter(total=len(video), desc=f'{application.name}: {video_name}', unit='frames')
            inference_results = {}

            for fid, video_slice in enumerate(video):
                progress_bar.update()
                inference_results[fid] = application.inference(video_slice.cuda(), detach=True)[0]

            write_results(video_name, application.name, inference_results, logger)


if __name__ == '__main__':

    # set the format of the logger
    coloredlogs.install(fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", level='INFO')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', type=str, help='The video file names to obtain inference results.', required=True, nargs='+')
    parser.add_argument('--confidence_threshold', type=float, help='The confidence score threshold for calculating accuracy.', default=0.3)
    parser.add_argument('--iou_threshold', type=float, help='The IoU threshold for calculating accuracy in object detection.', default=0.3)

    args = parser.parse_args()

    main(args)
