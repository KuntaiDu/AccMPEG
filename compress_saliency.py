'''
    Compress the video through gradient-based optimization.
'''

from utils.results_utils import read_results
from utils.mask_utils import *
from utils.video_utils import read_videos, write_video, get_qp_from_name
from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision import io
from torch.distributions.bernoulli import Bernoulli
import argparse
import coloredlogs
import logging
import enlighten
import torchvision.transforms as T
from PIL import Image
import seaborn as sns
sns.set()


def main(args):

    # initialize
    logger = logging.getLogger('compress')
    handler = logging.NullHandler()
    logger.addHandler(handler)
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    videos = videos
    bws = [0, 1]
    qps = [get_qp_from_name(video_name) for video_name in video_names]

    # construct applications
    application = FasterRCNN_ResNet50_FPN()
    application.cuda()

    # construct the mask
    video_shape = videos[-1].shape
    num_frames = video_shape[0]
    mask_shape = [num_frames, 1, video_shape[2] //
                  args.tile_size, video_shape[3] // args.tile_size]
    assert num_frames % args.batch_size == 0
    sum_grad = torch.zeros(mask_shape)
    neg_grad_exp = (-sum_grad * args.learning_rate).exp()
    mask = torch.zeros(mask_shape)
    mask.requires_grad = True

    plt.clf()
    plt.figure(figsize=(16, 10))
    ground_truth_results = read_results(
        args.ground_truth, application.name, logger)

    for iteration in range(args.num_iterations):

        logger.info(f'Processing application {application.name}')
        progress_bar = enlighten.get_manager().counter(
            total=videos[-1].shape[0], desc=f'Iteration {iteration}: {application.name}', unit='frames')

        total_loss = []

        for batch_id, (video_slices, mask_slice) in enumerate(zip(zip(*[video.split(args.batch_size) for video in videos]), mask.split(args.batch_size))):

            progress_bar.update(incr=args.batch_size)

            # construct hybrid image
            masked_image = generate_masked_video(
                mask_slice, video_slices, bws, args)

            # calculate the loss
            loss = application.calc_loss(
                masked_image.cuda(), [ground_truth_results[batch_id * args.batch_size + i] for i in range(args.batch_size)], args)
            loss.backward(retain_graph=True)
            total_loss.append(loss.item())

        # update mask through normalized EG
        mask.requires_grad = False
        sum_grad += mask.grad
        neg_grad_exp = -sum_grad * args.learning_rate
        mask = torch.where(neg_grad_exp > percentile(neg_grad_exp, 100-args.tile_percentage),
                           torch.ones_like(neg_grad_exp),
                           torch.zeros_like(neg_grad_exp))
        mask.requires_grad = True

        logger.info('App loss: %0.3f' % torch.tensor(
            total_loss).mean())

    # visualization
    for batch_id, (video_slices, mask_slice) in enumerate(zip(zip(*[video.split(args.batch_size) for video in videos]), mask.split(args.batch_size))):
        
        if batch_id % 30 == 0:
            logger.info('Visualizing frame %d.', batch_id)
            fid = batch_id * args.batch_size
            heat = tile_mask(mask[fid:fid+1, :, :, :],
                                args.tile_size)[0, 0, :, :]
            plt.clf()
            ax = sns.heatmap(heat.detach().numpy(), zorder=3, alpha=0.5)
            image = T.ToPILImage()(video_slices[-1][0, :, :, :])
            image = application.plot_results_on(
                ground_truth_results[fid], image, (255, 255, 255), args)
            # image = application.plot_results_on(
            #     None, image, (0, 255, 255), args)
            ax.imshow(image, zorder=3, alpha=0.5)
            Path(
                f'visualize/{args.output}/').mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f'visualize/{args.output}/{fid}_attn.png', bbox_inches='tight')

        

    # optimization done. No more gradients required.
    mask.requires_grad = False
    write_masked_video(mask, args, qps, bws, logger)
    # masked_video = generate_masked_video(mask, videos, bws, args)
    # write_video(masked_video, args.output, logger)


if __name__ == '__main__':

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", level='INFO')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', nargs='+',
                        help='The video file names. The largest video file will be the ground truth.', required=True)
    parser.add_argument('-s', '--source', type=str,
                        help='The original video source.', required=True)
    parser.add_argument('-g', '--ground_truth', type=str,
                        help='The ground truth results.', required=True)
    parser.add_argument('-o', '--output', type=str,
                        help='The output name.', required=True)
    parser.add_argument('--confidence_threshold', type=float,
                        help='The confidence score threshold for calculating accuracy.', default=0.5)
    parser.add_argument('--iou_threshold', type=float,
                        help='The IoU threshold for calculating accuracy in object detection.', default=0.5)
    parser.add_argument('--num_iterations', type=int,
                        help='Number of iterations for optimizing the mask.', default=10)
    parser.add_argument('--tile_size', type=int,
                        help='The tile size of the mask.', default=40)
    parser.add_argument('--learning_rate', type=float,
                        help='The learning rate.', default=100)
    parser.add_argument('--batch_size', type=int,
                        help='The batch size', default=1)
    parser.add_argument('--tile_percentage', type=float,
                        help='How many percentage of tiles will remain', default=5)
    # parser.add_argument('--mask_p', type=int, help='The p-norm for the mask.', default=1)
    # parser.add_argument('--binarize_weight', type=float, help='The weight of the mask binarization loss term.', default=1)
    # parser.add_argument('--cont_weight', type=float, help='The weight of the continuity normalization term', default=0)
    # parser.add_argument('--cont_p', type=int, help='The p-norm for the continuity.', default=1)

    args = parser.parse_args()

    main(args)
