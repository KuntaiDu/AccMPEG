
import torch
from torchvision import io
import argparse
import coloredlogs
import logging
import enlighten

from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN

# a video is by default a 4-D Tensor [Time, Height, Width, Channel]

def main(args):

    logger = logging.getLogger('main')
    handler = logging.NullHandler()
    logger.addHandler(handler)
    
    # read the video frames, and convert to [0,1]-range
    logger.info('Reading video')
    video = io.read_video(args.input, pts_unit='sec')[0].float().div(255).permute(0, 3, 1, 2)
    logger.info('Reading ground truth')
    gt = io.read_video(args.ground_truth, pts_unit='sec')[0].float().div(255).permute(0, 3, 1, 2)

    assert video.shape[0] == gt.shape[0], 'Video and ground-truth video should have the same number of frames.'

    application_bundle = [FasterRCNN_ResNet50_FPN()]
    application_accuracies = []

    for application in application_bundle:

        logger.info(f'Processing application {application.name()}')
        progress_bar = enlighten.get_manager().counter(total=gt.shape[0], desc=application.name(), unit='frames')

        application.cuda()
        accuracies = []

        for video_slice, gt_slice in zip(video.split(1), gt.split(1)):

            # inference on every frame
            accuracy = application.calc_accuracy(video_slice.cuda(), gt_slice.cuda(), args)
            accuracies.append(accuracy)
            progress_bar.update()

        application_accuracies.append(torch.Tensor(accuracies).mean())

        application.cpu()


if __name__ == '__main__':

    # set the format of the logger
    coloredlogs.install(fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", level='INFO')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, help='The video file name.', required=True)
    parser.add_argument('-g', '--ground_truth', type=str, help='The file name of ground-truth video.', required=True)
    parser.add_argument('--confidence_threshold', type=float, help='The confidence score threshold for calculating accuracy.', default=0.3)
    parser.add_argument('--iou_threshold', type=float, help='The IoU threshold for calculating accuracy in object detection.', default=0.3)

    args = parser.parse_args()

    main(args)
