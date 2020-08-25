
import torch
from torchvision import io
import argparse
import coloredlogs

from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN

# a video is by default a 4-D Tensor [Time, Height, Width, Channel]

def main(args):
    
    # read the video frames, and convert to [0,1]-range
    video = io.read_video(args.input, pts_unit='sec')[0].float().div(255).permute(0, 3, 1, 2)
    gt = io.read_video(args.ground-truth, pts_unit='sec')[0].float(0).div(255).permute(0, 3, 1, 2)

    application_bundle = [FasterRCNN_ResNet50_FPN()]

    # typically one GPU could support two 
    for application in application_bundle:

        application.cuda()

        for video_slice, gt_slice in zip(video.split(1), gt.split(1)):

            # inference on every frame

            inference_results_slice = application.inference(video_tensor_slice.cuda(), False)
            import pdb; pdb.set_trace()

        

    # run inference


if __name__ == '__main__':

    # set the format of the logger
    coloredlogs.install(fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", level='INFO')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, help='The video file name.')
    parser.add_argument('-g', '--ground-truth', type=str, help='The file name of ground-truth video.')

    args = parser.parse_args()

    main(args)
