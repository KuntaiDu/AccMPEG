'''
    Compress the video through gradient-based optimization.
'''

from utils.results_utils import read_results
from utils.mask_utils import *
from utils.video_utils import read_videos, write_video, get_qp_from_name
from utils.bbox_utils import center_size
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
    logger = logging.getLogger('black')
    handler = logging.NullHandler()
    logger.addHandler(handler)
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.input, logger, sort=True)
    videos = videos
    bws = [0, 1]
    qps = [get_qp_from_name(video_name) for video_name in video_names]

    # construct applications
    application = FasterRCNN_ResNet50_FPN()
    application.cuda()

    # construct the mask
    video_shape = [len(videos[-1]), 3, 720, 1280]
    num_frames = video_shape[0]
    mask_shape = [num_frames, 1, video_shape[2] //
                  args.tile_size, video_shape[3] // args.tile_size]
    assert num_frames % args.batch_size == 0
    sum_grad = torch.zeros(mask_shape)
    sum_grad = sum_grad.cuda()
    mask = torch.ones(mask_shape)

    # construct background tensor
    background_shape = [args.batch_size, 3, 720, 1280]
    background = torch.ones(background_shape)
    background = background * \
        torch.tensor(application.model.transform.image_mean)[
            None, :, None, None]
    background = background.cuda()

    # read ground truth
    ground_truth_results = read_results(
        args.ground_truth, application.name, logger)
    # construct the regions
    regions = [center_size(application.filter_results(ground_truth_results[i], args.confidence_threshold)[
                           2]) for i in ground_truth_results.keys()]
    # with initial w, h = 0
    for region in regions:
        region[:, 2:] = 0

    plt.clf()
    plt.figure(figsize=(16, 10))

    for iteration in range(args.num_iterations):

        logger.info(f'Run {application.name}')
        progress_bar = enlighten.get_manager().counter(
            total=len(videos[-1]), desc=f'Iteration {iteration}: {application.name}', unit='frames')

        f1s = []
        means = []

        for batch_id, (video_slices, mask_slice) in enumerate(zip(zip(*videos), mask.split(args.batch_size))):

            progress_bar.update(incr=args.batch_size)

            video_slices = [video_slice.cuda() for video_slice in video_slices]
            video_slices = [background, video_slices[0]]

            mask_slice = generate_mask_from_regions(
                mask_slice, regions[batch_id], bws[0], args.tile_size)
            mask_slice = mask_slice.cuda()

            # calculate the F1 score, to see the encoding difference
            with torch.no_grad():
                masked_image = generate_masked_video(
                    mask_slice, video_slices, bws, args)
                video_results = application.inference(
                    masked_image.cuda(), True)[0]
                f1s.append(application.calc_accuracy({
                    batch_id: video_results
                }, {
                    batch_id: ground_truth_results[batch_id]
                }, args)['f1'])
                means.append(mask_slice.mean())
                index = application.get_undetected_ground_truth_index(
                    ground_truth_results[batch_id], video_results, args)

            if iteration == args.num_iterations - 1:
                regions[batch_id][index, 2:] = 0
            else:
                regions[batch_id][index, 2:] += args.delta

        logger.info('Accuracy: %0.3f' % torch.tensor(
            f1s).mean())
        #logger.info('Maskmean: %0.3f' % torch.tensor(means).mean())
        logger.info('Maskmean: %0.3f' % mask.mean())

    # apply the latest update
    for batch_id, mask_slice in enumerate(mask.split(args.batch_size)):
        generate_mask_from_regions(mask_slice, regions[batch_id], bws[0], args.tile_size)


    # visualization
    for batch_id, (video_slices, mask_slice) in enumerate(zip(zip(*videos), mask.split(args.batch_size))):

        if batch_id % 30 == 0:
            logger.info('Visualizing frame %d.', batch_id)
            fid = batch_id * args.batch_size
            heat = tile_mask(mask[fid:fid+1, :, :, :],
                             args.tile_size)[0, 0, :, :].cpu()
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
    write_black_bkgd_video(mask, args, qps, bws, logger)
    # encode_masked_video(args, max(qps), mask, logger)
    # masked_video = generate_masked_video(mask, videos, bws, args)
    # write_video(masked_video, args.output, logger)


if __name__ == '__main__':

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", level='INFO')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', nargs=1, type=str,
                        help='The video file name.', required=True)
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
                        help='Number of iterations for optimizing the mask.', default=30)
    parser.add_argument('--tile_size', type=int,
                        help='The tile size of the mask.', default=40)
    parser.add_argument('--learning_rate', type=float,
                        help='The learning rate.', default=100)
    parser.add_argument('--batch_size', type=int,
                        help='The batch size', default=1)
    parser.add_argument('--tile_percentage', type=float,
                        help='How many percentage of tiles will remain', default=5)
    parser.add_argument('--delta', type=float,
                        help='The delta to enlarge the region.', default=16)
    # parser.add_argument('--mask_p', type=int, help='The p-norm for the mask.', default=1)
    # parser.add_argument('--binarize_weight', type=float, help='The weight of the mask binarization loss term.', default=1)
    # parser.add_argument('--cont_weight', type=float, help='The weight of the continuity normalization term', default=0)
    # parser.add_argument('--cont_p', type=int, help='The p-norm for the continuity.', default=1)

    args = parser.parse_args()

    main(args)
