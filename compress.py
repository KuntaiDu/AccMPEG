
import torch
from torchvision import io
import argparse
import coloredlogs
import logging
import enlighten
import torchvision.transforms as T
from PIL import Image
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from utils.video_utils import read_videos, write_video
from utils.mask_utils import *


def main(args):

    # initialize
    logger = logging.getLogger('compress')
    handler = logging.NullHandler()
    logger.addHandler(handler)
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    videos = videos
    bws = bws
    
    # construct applications
    application_bundle = [FasterRCNN_ResNet50_FPN()]

    # construct the mask
    video_shape = videos[-1].shape
    mask_shape = [video_shape[0], 1, video_shape[2] // args.tile_size, video_shape[3] // args.tile_size]
    mask = torch.ones(mask_shape).float()
    mask.requires_grad = True

    optimizer = torch.optim.SGD([mask], lr=args.learning_rate)
    plt.clf()
    plt.figure(figsize=(16, 10))
    
    for iteration in range(args.num_iterations):

        optimizer.zero_grad()
        (args.norm_weight * args.tile_size * args.tile_size * mask.mean()).backward()


        for application in application_bundle:

            logger.info(f'Processing application {application.name}')
            progress_bar = enlighten.get_manager().counter(total=videos[-1].shape[0], desc=f'Iteration {iteration}: {application.name}', unit='frames')

            application.cuda()

            for fid, (video_slices, mask_slice) in enumerate(zip(zip(*[video.split(1) for video in videos]), mask.split(1))):

                progress_bar.update()

                # construct hybrid image
                mask_slice = tile_mask(mask_slice, args.tile_size)
                masked_image = generate_masked_image(mask_slice, video_slices, bws)

                # calculate the loss
                loss = application.calc_loss(masked_image.cuda(), video_slices[-1].cuda(), args)
                loss.backward(retain_graph=True)

                # visualization
                if fid % 20 == 0 and iteration % 5 == 0:
                    # T.ToPILImage()(video_slices[-1][0, :, :, :]).save('visualize/%010d.png' % fid)
                    heat = tile_mask(mask[fid:fid+1, :, :, :], args.tile_size)[0, 0, :, :]
                    plt.clf()
                    sns.heatmap(heat.detach().numpy(), cmap = 'Blues_r')
                    plt.savefig('visualize/%010d-attn.png' % fid, bbox_inches='tight')
                

            application.cpu()

        optimizer.step()

        # clip mask to [0, 1]-range
        mask_clip(mask, bws[0])

    # optimization done. No more gradients required.
    mask.requires_grad = False
    # "binarize" the mask
    binarize_mask(mask, bws)
    # generate the compressed video based on the mask
    masked_video = generate_masked_video(mask, videos, bws, args)
    # import pdb; pdb.set_trace()
    write_video(masked_video, 'test_compressed.mp4', logger)
    # generate estimated bandwidth
    logger.info(f'The estimated normalized bandwidth is {mask.mean().item()}')



if __name__ == '__main__':

    # set the format of the logger
    coloredlogs.install(fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", level='INFO')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', nargs = '+', help='The video file names. The largest video file will be the ground truth.', required=True)
    parser.add_argument('--confidence_threshold', type=float, help='The confidence score threshold for calculating accuracy.', default=0.3)
    parser.add_argument('--num_iterations', type=int, help='Number of iterations for optimizing the mask.', default=50)
    parser.add_argument('--tile_size', type=int, help='The tile size of the mask.', default=16)
    parser.add_argument('--learning_rate', type=float, help='The learning rate.', default=4)
    parser.add_argument('--norm_weight', type=float, help='The weight of the l1 normalization term', default=4)

    args = parser.parse_args()

    main(args)
